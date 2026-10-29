#!/usr/bin/env node
// 唱歌 / AI 翻唱 MCP server（stdio）。
//
// 链路的完整形态：
//   联网搜歌 → 下载音轨 → 截片段 → demucs 分离人声/伴奏 → RVC 换成目标音色 → 混回伴奏 → 输出 mp3
//
// 三个工具：
//   sing_probe   环境自检（缺什么一次列全，不静默降级）
//   sing_search  搜歌，返回候选（B站优先，YouTube 兜底）
//   sing_cover   出成品，mode=original 只截原唱，mode=ai 走完整翻唱链路
//
// 设计约束：
// - 只写 <singDir>/work 与 outputDir，不碰其它文件。
// - 下载一律走 yt-dlp，不自己拼流媒体地址。
// - 缺依赖就明确报错并指出缺哪一项，绝不"悄悄地"退回原唱。
// - 产物文件名统一 sing- 前缀 + 清理过的标题，方便下游识别。
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { loadConfig } from './config.js';

const exec = promisify(execFile);

const CFG = loadConfig();
const SING_DIR = CFG.singDir;
const WORK_DIR = CFG.workDir;
const VENV_PY = CFG.pythonPath;
const RVC_DIR = CFG.rvcDir;
const COVER_SCRIPT = CFG.coverScript;
const F0_SCRIPT = CFG.f0Script;
const START_OFFSETS_FILE = CFG.startOffsetsFile;
const AI_VOICE_MODEL = CFG.pthModel;
const DEMUCS_MODEL = process.env.SING_DEMUCS_MODEL || 'htdemucs';
const YTDLP = CFG.ytdlpPath;
const FFMPEG = CFG.ffmpegPath;
const DEVICE = CFG.device;
const MAX_SECONDS = CFG.maxSeconds;
const DEFAULT_SECONDS = Math.min(CFG.defaultSeconds, MAX_SECONDS);
const TARGET_F0 = CFG.targetF0;
const FEMALE_F0 = CFG.femaleF0Threshold;
const MAX_PITCH = CFG.maxPitch;
const F0_ENGINE = CFG.f0Engine;
const F0_MAX_SECONDS = CFG.f0MaxSeconds;
const VOICE_TOOL = CFG.voiceToolName;
const MAX_CONCURRENT = CFG.maxConcurrent;
const KEEP_WORK = CFG.keepWork;
const KEEP_OUTPUT = CFG.keepOutput;
const CONTENTVEC_CANDIDATES = CFG.contentvecCandidates;
const RMVPE_MODEL = CFG.rmvpeModel;
const RVC_RUNTIME_DIR = CFG.rvcRuntimeDir;

// demucs 权重一旦缓存到本地，huggingface_hub 每次启动仍会联网校验版本：
// 网络不好时要么白等几十秒，要么直接失败。缓存存在就强制离线，只用本地权重。
// （换机器/换模型时把这两个环境变量 unset 即可恢复联网拉取。）
try {
  const torchCache = path.join(process.env.USERPROFILE || process.env.HOME || '', '.cache', 'torch', 'hub', 'checkpoints');
  if (fs.existsSync(torchCache) && fs.readdirSync(torchCache).some((f) => f.endsWith('.th'))) {
    process.env.HF_HUB_OFFLINE = '1';
    process.env.TRANSFORMERS_OFFLINE = '1';
  }
} catch { /* 读不到缓存目录就不设，行为和联网模式一致 */ }

async function run(file, args, opts = {}) {
  try {
    const { stdout } = await exec(file, args, {
      timeout: opts.timeout ?? 120_000,
      maxBuffer: 32 * 1024 * 1024,
      windowsHide: true,
      ...opts,
    });
    return String(stdout ?? '');
  } catch (error) {
    const detail = String(error?.stderr ?? '').trim().split(/\r?\n/).slice(-3).join(' | ');
    throw new Error(`${path.basename(file)} 执行失败：${error?.message ?? error}${detail ? `（${detail}）` : ''}`);
  }
}

/** 标题/文件名清理：NFKC 归一化（全角→半角）后只保留中文、ASCII 字母数字与常见符号。
 *  必须滤掉 𝐇𝐢 这类数学粗体字符：它们在视觉上像字母，但会让文件名在部分客户端里乱码。 */
function slugify(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .replace(/[\\/:*?"<>|\r\n\t]+/g, ' ')
    .replace(/[^\p{Script=Han}A-Za-z0-9._ -]+/gu, ' ')
    .replace(/\s+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40) || 'song';
}

function stamp() {
  return new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
}

/** 能不能真的跑起来。
 *  [坑] 早先用 fs.existsSync(ffmpegPath) 判断就绪：PATH 兜底时 ffmpegPath 只是
 *  裸名字 "ffmpeg.exe"，existsSync 按 cwd 解析必然 false —— 明明能用却报缺失。
 *  这里改成真的 spawn 一次，退出码非零但有输出也算可用（-version 类命令就这样）。 */
async function canRun(file, args = ['--version']) {
  try {
    await exec(file, args, { timeout: 30_000, windowsHide: true, maxBuffer: 4 * 1024 * 1024 });
    return { ok: true };
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return { ok: false, detail: '找不到可执行文件（既不在配置里，也不在 PATH 上）' };
    }
    const text = `${String(error?.stdout ?? '')}${String(error?.stderr ?? '')}`.trim();
    if (text) return { ok: true, detail: text.split(/\r?\n/)[0].slice(0, 80) };
    return { ok: false, detail: String(error?.message ?? error).slice(0, 160) };
  }
}

/** 目录存在且真的能写。只判 existsSync 会漏掉「目录在但没权限」这种最常见的情况。 */
function canWriteDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true });
    const probe = path.join(dir, `.write-probe-${process.pid}-${Date.now()}`);
    fs.writeFileSync(probe, 'ok');
    fs.unlinkSync(probe);
    return { ok: true };
  } catch (error) {
    return { ok: false, detail: String(error?.message ?? error).slice(0, 160) };
  }
}

function firstExistingPath(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return candidates[0] ?? '';
}

/** 输出目录里不重名的目标路径。 */
function uniqueOutPath(name) {
  fs.mkdirSync(CFG.outputDir, { recursive: true });
  const ext = path.extname(name);
  const base = name.slice(0, name.length - ext.length);
  let candidate = path.join(CFG.outputDir, name);
  let i = 2;
  while (fs.existsSync(candidate)) {
    candidate = path.join(CFG.outputDir, `${base}-${i}${ext}`);
    i += 1;
  }
  return candidate;
}

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36';

function stripTags(text) {
  return String(text ?? '').replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&quot;/g, '"').trim();
}

/** "1:57" / "61:8" / "1:02:03" → 秒。 */
function parseClock(text) {
  const parts = String(text ?? '').split(':');
  if (!parts.length || parts.some((p) => p === '' || Number.isNaN(Number(p)))) return undefined;
  return parts.reduce((acc, p) => acc * 60 + Number(p), 0);
}

/**
 * B站公开搜索接口（不需要登录）。
 * 坑：不带 cookie 会被风控直接 412。先访问首页取 buvid3 再带上去即可；
 * 命中 412 时清掉缓存重取一次（buvid 会过期）。
 */
let biliCookie = null;

async function ensureBiliCookie() {
  if (biliCookie !== null) return biliCookie;
  try {
    const res = await fetch('https://www.bilibili.com/', {
      headers: { 'user-agent': UA },
      signal: AbortSignal.timeout(15_000),
    });
    const raw = typeof res.headers.getSetCookie === 'function' ? res.headers.getSetCookie() : [];
    biliCookie = raw
      .map((c) => c.split(';')[0])
      .filter((c) => /buvid|b_nut|_uuid/i.test(c))
      .join('; ');
  } catch {
    biliCookie = '';
  }
  return biliCookie;
}

async function searchBilibili(query, limit) {
  const url = `https://api.bilibili.com/x/web-interface/search/type?search_type=video&keyword=${encodeURIComponent(query)}&page=1`;
  let lastError = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    const cookie = await ensureBiliCookie();
    const res = await fetch(url, {
      headers: {
        'user-agent': UA,
        referer: 'https://www.bilibili.com/',
        ...(cookie ? { cookie } : {}),
      },
      signal: AbortSignal.timeout(20_000),
    });
    if (res.status === 412 && attempt === 0) {
      biliCookie = null; // 风控或 cookie 过期：重取一次再试
      lastError = new Error('B站搜索接口 HTTP 412（已重取 cookie 重试）');
      continue;
    }
    if (!res.ok) throw new Error(`B站搜索接口 HTTP ${res.status}`);
    const data = await res.json();
    if (data?.code !== 0) throw new Error(`B站搜索接口 code=${data?.code} ${data?.message ?? ''}`);
    return (data?.data?.result ?? [])
      .slice(0, limit)
      .map((item) => ({
        title: stripTags(item.title),
        url: `https://www.bilibili.com/video/${item.bvid}`,
        duration: parseClock(item.duration),
        // 播放量是判断「哪一版才是大众认知的那首」最省事的信号：
        // 搜索结果里混着翻唱、切片、直播回放，光看标题区分不出来
        play: Number.isFinite(Number(item.play)) ? Number(item.play) : undefined,
        uploader: item.author ? stripTags(item.author) : undefined,
        source: 'bilibili',
      }))
      .filter((item) => item.title && item.url);
  }
  throw lastError ?? new Error('B站搜索接口失败');
}

async function searchSongs(query, limit = 5) {
  const errors = [];
  try {
    const hits = await searchBilibili(query, limit);
    if (hits.length) return hits;
  } catch (error) {
    errors.push(`bilibili-api: ${error?.message ?? error}`);
  }
  const attempts = [
    { prefix: `bilisearch${limit}:`, source: 'bilibili' },
    { prefix: `ytsearch${limit}:`, source: 'youtube' },
  ];
  for (const attempt of attempts) {
    try {
      // 不用 --flat-playlist：B站搜索的扁平结果只有链接、没有标题和时长。
      const out = await run(YTDLP, [
        '--no-warnings', '--ignore-errors', '-J',
        `${attempt.prefix}${query}`,
      ], { timeout: 120_000 });
      const doc = JSON.parse(out);
      const entries = (doc.entries ?? []).filter(Boolean).map((e) => ({
        title: String(e.fulltitle ?? e.title ?? '').trim(),
        url: String(e.webpage_url ?? e.url ?? '').trim(),
        duration: Number.isFinite(Number(e.duration)) ? Number(e.duration) : undefined,
        uploader: e.uploader ? String(e.uploader) : undefined,
        source: attempt.source,
      })).filter((e) => e.url);
      if (entries.length) return entries;
    } catch (error) {
      errors.push(`${attempt.source}: ${error?.message ?? error}`);
    }
  }
  throw new Error(`搜索失败：${errors.join(' / ') || '没有结果'}`);
}

/** 下载最佳音轨并转成 mp3（只保留音频，不带视频）。
 *  [坑] --ffmpeg-location 只在拿到绝对路径时传。PATH 兜底时 FFMPEG 只是裸名字，
 *  path.dirname('ffmpeg.exe') 得到 "."，等于骗 yt-dlp 去当前目录找 ffmpeg。 */
async function downloadAudio(url, outBase) {
  await fsp.mkdir(WORK_DIR, { recursive: true });
  await run(YTDLP, [
    '--no-warnings', '--ignore-errors',
    '-f', 'bestaudio/best',
    '-x', '--audio-format', 'mp3', '--audio-quality', '5',
    ...(path.isAbsolute(FFMPEG) ? ['--ffmpeg-location', path.dirname(FFMPEG)] : []),
    '-o', `${outBase}.%(ext)s`,
    '--no-playlist',
    url,
  ], { timeout: 300_000 });
  for (const ext of ['mp3', 'm4a', 'webm', 'opus', 'aac', 'wav']) {
    const candidate = `${outBase}.${ext}`;
    if (fs.existsSync(candidate) && fs.statSync(candidate).size > 1024) return candidate;
  }
  throw new Error('下载完成但没找到音频文件');
}

/** 截一段。[start, start+seconds)，统一转 44.1k 立体声。 */
async function cutSegment(srcFile, outFile, startSeconds, seconds) {
  await fsp.mkdir(path.dirname(outFile), { recursive: true });
  await run(FFMPEG, [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-ss', String(Math.max(0, startSeconds)),
    '-t', String(seconds),
    '-i', srcFile,
    '-ar', '44100', '-ac', '2', '-b:a', '192k',
    outFile,
  ], { timeout: 120_000 });
  if (!fs.existsSync(outFile) || fs.statSync(outFile).size < 1024) throw new Error('裁剪失败：输出为空');
  return outFile;
}

/** 人声/伴奏分离（demucs）。返回 { vocals, instrumental }。 */
async function separateVocals(inputWav, workDir) {
  const outDir = path.join(workDir, 'demucs');
  await run(VENV_PY, ['-m', 'demucs', '-n', DEMUCS_MODEL, '--two-stems=vocals', '-d', DEVICE, '-o', outDir, inputWav], { timeout: 900_000 });
  const base = path.basename(inputWav, path.extname(inputWav));
  const vocals = path.join(outDir, DEMUCS_MODEL, base, 'vocals.wav');
  const instrumental = path.join(outDir, DEMUCS_MODEL, base, 'no_vocals.wav');
  if (!fs.existsSync(vocals) || !fs.existsSync(instrumental)) throw new Error('人声分离失败：没有生成 vocals / no_vocals');
  return { vocals, instrumental };
}

/** 用目标音色跑 RVC 转换。 */
async function convertVoice(inputWav, outWav, pitch = 0) {
  if (!fs.existsSync(COVER_SCRIPT)) throw new Error(`缺少 RVC 驱动脚本：${COVER_SCRIPT}`);
  const log = await run(VENV_PY, [COVER_SCRIPT, inputWav, outWav, String(pitch), 'rmvpe'], { timeout: 900_000 });
  if (!fs.existsSync(outWav) || fs.statSync(outWav).size < 1024) {
    throw new Error(`RVC 转换失败：${String(log).slice(-300)}`);
  }
  return outWav;
}

/** 估原唱人声的中位基频（Hz）：>=200 视为女声区，<165 视为男声区。
 *  升调只能由「原唱音域低于目标音色」触发，不能凭感觉乱加——加错了会顶到破音区。
 *  引擎可选：auto 优先用 rmvpe（与 cover.py 转换阶段同一条 f0 曲线，也更快），
 *  探测不到就由 est_f0.py 内部退回 pyin；返回里带上是哪套算的，便于对比排查。 */
async function estimateMedianF0(wav) {
  try {
    if (!fs.existsSync(F0_SCRIPT)) return null;
    const args = [F0_SCRIPT, wav];
    if (F0_ENGINE) args.push(`--engine=${F0_ENGINE}`);
    if (F0_MAX_SECONDS) args.push(`--max-seconds=${F0_MAX_SECONDS}`);
    const out = String(await run(VENV_PY, args, { timeout: 180_000 }) ?? '');
    const m = /MEDIAN_F0=([\d.]+)/.exec(out);
    const e = /ENGINE=([A-Za-z0-9_]+)/.exec(out);
    const hz = m ? Number.parseFloat(m[1]) : 0;
    const engine = e ? e[1] : null;
    if (engine) console.error(`[mcp-sing] est_f0 engine=${engine}${engine === 'none' ? ` (${/REASON=([^\n]+)/.exec(out)?.[1] ?? ''})` : ''}`);
    return Number.isFinite(hz) && hz > 0 ? { hz, engine } : null;
  } catch {
    return null;
  }
}

/** 按歌名记住的起唱点与时长：命中就强制覆盖（不让模型自己估前奏长度）。
 *  规则值可以是数字（只定起点），也可以是 { start, seconds }（起点 + 这首歌固定多长）。 */
function resolveStartOffset(song, pickedTitle, requested) {
  try {
    if (!fs.existsSync(START_OFFSETS_FILE)) return { seconds: requested ?? 0, duration: null, key: null };
    const table = JSON.parse(fs.readFileSync(START_OFFSETS_FILE, 'utf8'));
    if (!table || typeof table !== 'object') return { seconds: requested ?? 0, duration: null, key: null };
    const hay = `${song ?? ''} ${pickedTitle ?? ''}`.toLowerCase();
    for (const [k, v] of Object.entries(table)) {
      if (typeof k !== 'string' || !k.trim() || k.startsWith('_')) continue;
      if (!hay.includes(k.toLowerCase())) continue;
      if (v && typeof v === 'object') {
        const start = Number(v.start);
        const dur = Number(v.seconds);
        if (Number.isFinite(start) && start >= 0) {
          return { seconds: start, duration: Number.isFinite(dur) && dur > 0 ? dur : null, key: k };
        }
        continue;
      }
      const sec = Number(v);
      if (Number.isFinite(sec) && sec >= 0) return { seconds: sec, duration: null, key: k };
    }
  } catch { /* 规则文件坏了就退回请求里给的值，不影响出歌 */ }
  return { seconds: requested ?? 0, duration: null, key: null };
}

/** 把干声和伴奏混成一条成品。
 *  [电平坑] amix 默认 normalize=1 会把总和除以输入路数（-6dB），于是成品比源截取低 10dB 左右，
 *  前奏段只剩被砍半的伴奏，听感像「前面卡顿、没声音」。
 *  这里先试带 normalize=0 的新版 ffmpeg（电平最准），老版本不认这个参数时回退，
 *  用 volume=2 补回那 6dB，再 loudnorm 统一响度、alimiter 防削顶。 */
async function mixTracks(vocalsWav, instrumentalWav, outFile, seconds) {
  await fsp.mkdir(path.dirname(outFile), { recursive: true });
  const base = '[0:a]volume=1.0[v];[1:a]volume=0.72[i];[v][i]amix=inputs=2:duration=longest:dropout_transition=0';
  const attempts = [
    `${base}:normalize=0,loudnorm=I=-15:TP=-1.5:LRA=11,alimiter=limit=0.95[out]`,
    `${base},volume=2,loudnorm=I=-15:TP=-1.5:LRA=11,alimiter=limit=0.95[out]`,
  ];
  let lastError = null;
  for (const filter of attempts) {
    try {
      await run(FFMPEG, [
        '-hide_banner', '-loglevel', 'error', '-y',
        '-i', vocalsWav, '-i', instrumentalWav,
        '-filter_complex', filter,
        '-map', '[out]', '-t', String(seconds),
        '-ar', '44100', '-ac', '2', '-b:a', '192k', outFile,
      ], { timeout: 180_000 });
      if (fs.existsSync(outFile) && fs.statSync(outFile).size > 1024) return outFile;
      lastError = new Error('混音输出为空');
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error('混音失败');
}

/** 读音频时长（秒），失败返回 null。
 *  用 ffmpeg -i（不带输出）：它会以非零码退出、把时长写在 stderr 里，所以不能走 run()，
 *  否则 stderr 在包装错误时被丢掉，永远解析不到。 */
async function probeDuration(file) {
  try {
    const stderr = await exec(FFMPEG, ['-hide_banner', '-i', file], {
      timeout: 60_000,
      windowsHide: true,
    }).then(
      (r) => String(r.stderr ?? ''),
      (e) => String(e?.stderr ?? ''),
    );
    const m = /Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)/.exec(stderr);
    if (!m) return null;
    return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]);
  } catch {
    return null;
  }
}

/** 环境自检：每个环节缺什么一眼能看到。
 *  两件事必须做到，否则「缺什么一次列全」就是空话：
 *   1. 判定方式要和真正执行时一致（PATH 上的可执行文件用 spawn 判，不用 existsSync）；
 *   2. 覆盖要和对 ai 模式的准入门槛完全重合（不然 probe 全绿、sing_cover 却缺 contentvec）。
 *  自检本身要 spawn 两次可执行文件 + 两次 python，约 2~4s；每次点歌都跑一遍太浪费，
 *  所以结果缓存 30 秒，sing_probe 显式要求 fresh 时不走缓存。 */
let readinessCache = null;

async function readiness({ fresh = false } = {}) {
  if (!fresh && readinessCache && Date.now() - readinessCache.at < 30_000) return readinessCache.value;
  const checks = {
    ytdlp: { ...(await canRun(YTDLP, ['--version'])), path: YTDLP },
    ffmpeg: { ...(await canRun(FFMPEG, ['-version'])), path: FFMPEG },
    venv: { ok: fs.existsSync(VENV_PY), path: VENV_PY },
    torch: { ok: false, detail: '未检测' },
    demucs: { ok: false, detail: '未检测' },
    rvc: {
      ok: fs.existsSync(path.join(RVC_DIR, 'rvc')) && fs.existsSync(path.join(RVC_DIR, 'rvc_cli')),
      path: RVC_DIR,
      detail: fs.existsSync(RVC_DIR) ? '需要同时有 rvc/ 与 rvc_cli/ 两个子目录' : '目录不存在',
    },
    coverScript: { ok: fs.existsSync(COVER_SCRIPT), path: COVER_SCRIPT },
    f0Script: { ok: fs.existsSync(F0_SCRIPT), path: F0_SCRIPT },
    voiceModel: { ok: fs.existsSync(AI_VOICE_MODEL), path: AI_VOICE_MODEL },
    contentvec: { ok: fs.existsSync(firstExistingPath(CONTENTVEC_CANDIDATES)), path: firstExistingPath(CONTENTVEC_CANDIDATES) },
    rmvpe: {
      ok: fs.existsSync(RMVPE_MODEL) || fs.existsSync(path.join(RVC_DIR, 'rvc', 'models', 'predictors', 'rmvpe.pt')),
      path: RMVPE_MODEL,
      detail: '放 sing/model/rmvpe.pt，首次运行会自动复制进 rvc_cli',
    },
    rvcRuntime: { ok: fs.existsSync(RVC_RUNTIME_DIR), path: RVC_RUNTIME_DIR },
    index: { ok: true, detail: '未检测' },
    startOffsetsFile: {
      ok: fs.existsSync(START_OFFSETS_FILE),
      path: START_OFFSETS_FILE,
      detail: '可选的起唱点规则文件（不存在不影响出歌，只是每条都要自己估前奏）',
    },
  };
  if (checks.venv.ok) {
    try {
      const out = await run(VENV_PY, ['-c', 'import torch,sys;print(torch.__version__, torch.cuda.is_available())'], { timeout: 60_000 });
      checks.torch = { ok: true, detail: out.trim() };
    } catch (error) {
      checks.torch = { ok: false, detail: String(error?.message ?? error).slice(0, 200) };
    }
    try {
      await run(VENV_PY, ['-c', 'import demucs'], { timeout: 60_000 });
      checks.demucs = { ok: true };
    } catch {
      checks.demucs = { ok: false, detail: '未安装（pip install demucs）' };
    }
  }
  try {
    const modelDir = path.dirname(AI_VOICE_MODEL);
    const idx = fs.existsSync(modelDir) ? fs.readdirSync(modelDir).filter((f) => f.toLowerCase().endsWith('.index')) : [];
    checks.index = idx.length
      ? { ok: true, detail: `${idx.length} 个索引，将自动以 index_rate=0.75 启用：${idx.slice(0, 3).join(', ')}` }
      : { ok: true, detail: '无 faiss 索引（纯模型推理，音色相似度略低）' };
  } catch {
    checks.index = { ok: true, detail: '模型目录读不到，跳过' };
  }
  checks.outputDir = { ...canWriteDir(CFG.outputDir), path: CFG.outputDir };
  checks.device = { value: DEVICE };
  // ai 模式的准入门槛：与 sing_cover 的前置检查共用同一份清单，避免两处判据漂移
  checks.aiMissing = AI_REQUIRED.filter(([key]) => checks[key] && checks[key].ok === false)
    .map(([, label]) => label);
  checks.aiReady = checks.aiMissing.length === 0;
  checks.originalReady = checks.ytdlp.ok && checks.ffmpeg.ok && checks.outputDir.ok;
  readinessCache = { at: Date.now(), value: checks };
  return checks;
}

/** ai 模式真正需要的东西：键 → 给人看的说法。 */
const AI_REQUIRED = [
  ['venv', 'Python venv'],
  ['torch', 'PyTorch'],
  ['demucs', 'demucs（人声分离）'],
  ['rvc', 'RVC 推理代码'],
  ['coverScript', 'RVC 驱动脚本 sing/cover.py'],
  ['voiceModel', '音色模型 .pth'],
  ['contentvec', 'contentvec 特征模型'],
  ['rmvpe', 'rmvpe 音高模型'],
];

/** 清理：work 下只留最近 N 个任务目录、output 下只留最近 N 个成品。
 *  机器人是真 7x24 跑的，一首歌的中间产物 50~150 MB，不清理一个月就是几百 GB。 */
function pruneWork(keep = KEEP_WORK) {
  if (!keep) return { removed: 0 };
  try {
    const dirs = fs.readdirSync(WORK_DIR)
      .map((name) => path.join(WORK_DIR, name))
      .filter((p) => {
        try { return fs.statSync(p).isDirectory(); } catch { return false; }
      })
      .map((p) => ({ p, t: fs.statSync(p).mtimeMs }))
      .sort((a, b) => b.t - a.t);
    let removed = 0;
    for (const item of dirs.slice(keep)) {
      try { fs.rmSync(item.p, { recursive: true, force: true }); removed += 1; } catch { /* 占用中就下次 */ }
    }
    return { removed };
  } catch {
    return { removed: 0 };
  }
}

function pruneOutput(keep = KEEP_OUTPUT) {
  if (!keep) return { removed: 0 };
  try {
    const files = fs.readdirSync(CFG.outputDir)
      .filter((name) => name.startsWith('sing-') && /\.(mp3|wav|m4a)$/i.test(name))
      .map((name) => ({ p: path.join(CFG.outputDir, name), t: fs.statSync(path.join(CFG.outputDir, name)).mtimeMs }))
      .sort((a, b) => b.t - a.t);
    let removed = 0;
    for (const item of files.slice(keep)) {
      try { fs.unlinkSync(item.p); removed += 1; } catch { /* 正在被读就跳过 */ }
    }
    return { removed };
  } catch {
    return { removed: 0 };
  }
}

/** 并发闸门：RVC 单次要吃几个 GB 显存，两个群同时点歌就是 CUDA OOM。 */
let activeJobs = 0;
const jobWaiters = [];
async function acquireSlot() {
  if (activeJobs < MAX_CONCURRENT) {
    activeJobs += 1;
    return;
  }
  await new Promise((resolve) => jobWaiters.push(resolve));
  activeJobs += 1;
}
function releaseSlot() {
  activeJobs = Math.max(0, activeJobs - 1);
  const next = jobWaiters.shift();
  if (next) next();
}

/** 长任务进度：客户端给了 progressToken 才发，没有就静默跳过。
 *  没有它，CPU 模式下几分钟的翻唱很容易被客户端当成「卡死」。 */
function progressReporter(extra) {
  const token = extra?._meta?.progressToken;
  const send = extra?.sendNotification;
  if (token === undefined || typeof send !== 'function') return () => {};
  let n = 0;
  return (message) => {
    n += 1;
    Promise.resolve(send({
      method: 'notifications/progress',
      params: { progressToken: token, progress: n, total: 6, message },
    })).catch(() => {});
  };
}

function json(value) {
  return { content: [{ type: 'text', text: JSON.stringify(value, null, 1) }] };
}

function err(text) {
  return { content: [{ type: 'text', text }], isError: true };
}

const server = new McpServer({ name: 'sing', version: '1.0.0' });

server.tool(
  'sing_probe',
  '检查唱歌链路是否就绪：yt-dlp / ffmpeg / torch(CUDA) / demucs / RVC 工程 / 音色模型 / contentvec / 输出目录可写。' +
    '返回里 aiReady 表示能不能真翻唱，aiMissing 是还缺什么，originalReady 表示能不能出原唱片段。要确认「现在能不能唱」时先调这个。',
  {},
  async () => {
    try {
      return json(await readiness({ fresh: true }));
    } catch (error) {
      return err(`自检失败：${error?.message ?? error}`);
    }
  },
);

server.tool(
  'sing_search',
  '在线搜歌（B站优先，YouTube 兜底），返回候选的标题/时长/播放量/链接(含上传者)。' +
    '挑版本时优先选播放量高、时长 1.5~8 分钟的整曲；标着「切片/片段/翻唱/live」的往往是几十秒的短视频，' +
    '拿去翻唱会直接踩到副歌中间。不确定是哪一版时，先用它挑一首再调 sing_cover。',
  {
    query: z.string().min(1).max(80).describe('歌名，可带歌手（如「恋爱循环 花泽香菜」）'),
    limit: z.number().int().min(1).max(8).optional().describe('返回几条，默认 5'),
  },
  async ({ query, limit }) => {
    try {
      return json({ query, results: await searchSongs(query, limit ?? 5) });
    } catch (error) {
      return err(`搜歌失败：${error?.message ?? error}`);
    }
  },
);

server.tool(
  'sing_cover',
  `唱一段歌：联网找歌 → 下载音源 → 截最多 ${MAX_SECONDS} 秒 → 输出到产物目录，返回文件名（之后用机器人框架的发语音工具发出去）。` +
  'mode=original 是原唱片段；mode=ai 是用目标音色翻唱（需要本地 RVC 与音色模型就绪，没就绪会明确报缺什么，不会偷偷退回原唱）。' +
  'mode=ai 时不要自己猜 pitch：留空即可，脚本会按原唱音域自动定调，返回里的 pitchNote 会说明它做了什么。',
  {
    song: z.string().min(1).max(80).describe('歌名，可带歌手'),
    seconds: z.number().int().min(5).max(MAX_SECONDS).optional().describe(`截多长，最长 ${MAX_SECONDS} 秒，默认 ${DEFAULT_SECONDS}`),
    startSeconds: z.number().min(0).max(3600).optional().describe('从第几秒开始截（默认 0；想抓高潮段可以自己估一个）'),
    mode: z.enum(['original', 'ai']).default('original').describe('original=原唱片段（默认）；ai=用目标音色翻唱'),
    candidate: z.number().int().min(1).max(8).optional().describe('搜索结果里取第几个（默认 1；取第 N 个时会把搜索条数相应放宽，可用 sing_search 先看）'),
    pitch: z.number().int().min(-24).max(24).optional().describe(`变调（半音）。默认留空：脚本按原唱音域自动定调（上限 +${MAX_PITCH}）。只有明确知道原唱比目标音色低一个八度时才手填`),
  },
  async ({ song, seconds, startSeconds, mode, candidate, pitch }, extra) => {
    const wantAi = mode === 'ai';
    const report = progressReporter(extra);
    await acquireSlot();
    try {
      if (wantAi) {
        const checks = await readiness();
        if (checks.aiMissing.length) {
          return err(`AI 翻唱还没就绪，缺少：${checks.aiMissing.join('、')}。可以先用 mode=original 出原唱片段，或先把环境装完（详见 sing_probe 的 path 字段）。`);
        }
      }
      report('搜歌');
      let length = Math.min(seconds ?? DEFAULT_SECONDS, MAX_SECONDS);
      // candidate 最大 8，搜索条数必须跟上，否则 candidate=6~8 永远取不到
      const results = await searchSongs(song, Math.max(5, candidate ?? 1));
      const picked = results[Math.max(0, Math.min((candidate ?? 1) - 1, results.length - 1))];
      if (!picked) throw new Error('没有搜到可用结果');
      await fsp.mkdir(WORK_DIR, { recursive: true });
      const base = path.join(WORK_DIR, `${wantAi ? 'ai' : 'src'}-${stamp()}-${process.pid}-${Math.random().toString(36).slice(2, 6)}`);
      await fsp.mkdir(base, { recursive: true });
      report('下载音源');
      const downloaded = await downloadAudio(picked.url, path.join(base, 'src'));
      const total = await probeDuration(downloaded);
      // 起始点与时长：歌名命中用户规则就强制用它（不让模型再自己估前奏长度）
      const startRule = resolveStartOffset(song, picked.title, startSeconds);
      if (startRule.duration) length = Math.min(startRule.duration, MAX_SECONDS);
      const start = Math.min(startRule.seconds, Math.max(0, (total ?? 60) - length - 1));
      const outName = `sing-${slugify(picked.title || song)}${wantAi ? '-ai' : ''}-${stamp()}.mp3`;
      const outFile = uniqueOutPath(outName);

      if (!wantAi) {
        report('截取片段');
        await cutSegment(downloaded, outFile, start, length);
        await fsp.unlink(downloaded).catch(() => {});
        return json({
          name: path.basename(outFile),
          path: outFile,
          title: picked.title,
          source: picked.source,
          url: picked.url,
          requestedSeconds: length,
          startSeconds: start,
          startOffsetRule: startRule.key,
          sourceDurationSeconds: total,
          mode: 'original',
          bytes: fs.statSync(outFile).size,
          next: `用 ${VOICE_TOOL} 传 "${path.basename(outFile)}" 发出去`,
        });
      }

      // ── AI 翻唱：截片段 → 分离人声/伴奏 → 换音色 → 混回伴奏 ──
      report('截取片段');
      const clipWav = path.join(base, 'clip.wav');
      await cutSegment(downloaded, clipWav, start, length);
      await fsp.unlink(downloaded).catch(() => {});
      report('分离人声与伴奏');
      const { vocals, instrumental } = await separateVocals(clipWav, base);
      const converted = path.join(base, 'vocals-ai.wav');
      // ── 定调：把源人声搬到目标音色的舒适区，而不是机械升降八度 ──
      const medianF0 = await estimateMedianF0(vocals);
      let usePitch = pitch ?? 0;
      let pitchNote = null;
      if (medianF0 != null) {
        const hz = medianF0.hz.toFixed(0);
        if (medianF0.hz >= FEMALE_F0) {
          // 已经在女声区：原调就是目标音色的音域，任何升调都只会破音
          if (usePitch !== 0) {
            pitchNote = `原唱人声中位 ${hz}Hz 已高于 ${FEMALE_F0}Hz，把 pitch=${usePitch} 归 0`;
            usePitch = 0;
          }
        } else {
          // 目标音域法：把源中位 f0 搬到 TARGET_F0 附近。
          // 实测机械八度（+12）会顶到假声、丢帧率最高；搬到舒适区中心才是自然档。
          const sug = Math.max(0, Math.min(MAX_PITCH, Math.round(12 * Math.log2(TARGET_F0 / medianF0.hz))));
          if (usePitch !== sug) {
            pitchNote = `原唱人声中位 ${hz}Hz 低于目标音色音域，按目标 ${TARGET_F0}Hz 自动定调 pitch=${sug}` +
              (pitch !== undefined ? `（你填的 ${pitch} 已校正）` : '');
            usePitch = sug;
          }
        }
      }
      // 引擎差异在音域分界附近可能翻转判断：实测 pyin 与 rmvpe 在 245~291Hz 区间差 5~6Hz，
      // 若中位数正好压在 femaleF0Threshold 两侧，换引擎就会一边不升调、一边自动升调。
      // 这里只把情况说出来（让人能复核），不擅自改判。
      if (medianF0 != null && Math.abs(medianF0.hz - FEMALE_F0) <= 8) {
        const side = medianF0.hz >= FEMALE_F0 ? '不升调' : '自动定调';
        const boundary = `中位 ${medianF0.hz.toFixed(0)}Hz 距音域分界 ${FEMALE_F0}Hz 仅 ${Math.abs(medianF0.hz - FEMALE_F0).toFixed(0)}Hz，换引擎可能翻转判断，本次按${side}处理`;
        pitchNote = pitchNote ? `${pitchNote}；${boundary}` : boundary;
      }
      report(`用目标音色转换（pitch=${usePitch}）`);
      await convertVoice(vocals, converted, usePitch);
      report('混音与响度归一');
      await mixTracks(converted, instrumental, outFile, length);

      return json({
        name: path.basename(outFile),
        path: outFile,
        title: picked.title,
        source: picked.source,
        url: picked.url,
        requestedSeconds: length,
        startSeconds: start,
        startOffsetRule: startRule.key,
        sourceDurationSeconds: total,
        mode: 'ai',
        pitch: usePitch,
        sourceMedianF0: medianF0 == null ? null : Math.round(medianF0.hz),
        sourceF0Engine: medianF0?.engine ?? null,
        pitchNote,
        bytes: fs.statSync(outFile).size,
        next: `用 ${VOICE_TOOL} 传 "${path.basename(outFile)}" 发出去`,
      });
    } catch (error) {
      return err(`唱歌失败：${error?.message ?? error}`);
    } finally {
      releaseSlot();
      // 失败分支也要清：跑坏的 work 目录一样占几十上百 MB
      pruneWork();
      pruneOutput();
    }
  },
);

process.on('uncaughtException', (error) => {
  console.error('[mcp-sing] uncaught:', error?.message ?? error);
});
process.on('unhandledRejection', (error) => {
  console.error('[mcp-sing] unhandled:', error?.message ?? error);
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error(`[mcp-sing] ready · sing=${SING_DIR} · out=${CFG.outputDir} · device=${DEVICE}`);
