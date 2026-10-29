// 配置加载：环境变量 > config.json > 内置默认值。
//
// 设计原则：所有路径都相对「仓库根目录」解析，配置里可以写绝对路径，
// 但仓库本身不出现任何机器相关的绝对路径，换台机器 clone 下来即可用。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** 仓库根目录（本文件放在 <root>/src 下）。 */
export const ROOT = path.resolve(HERE, '..');

const DEFAULTS = {
  // 唱歌链路的工作目录：python 脚本、venv、模型都放这里
  singDir: 'sing',
  // 产物目录（生成的 mp3 落在这里，交给机器人框架去发送）
  outputDir: 'output',
  // 音色模型名，相对 <singDir>/model/
  model: 'xiaoxiao.pth',
  // 人声分离设备：cuda / cpu
  device: 'cuda',
  // 单次最长秒数（QQ 语音条本身还有 300 秒上限）
  maxSeconds: 60,
  // 不传 seconds 时的默认长度
  defaultSeconds: 30,
  // 定调目标：把原唱人声中位基频搬到这个值附近（Hz）
  targetF0: 250,
  // 中位基频高于此值视为「已经是女声区」，不再升调
  femaleF0Threshold: 200,
  // 自动定调上限（半音）
  maxPitch: 12,
  // 估调用的 f0 引擎：pyin（默认，与改动前行为一致）/ rmvpe / auto（优先 rmvpe，失败退回 pyin）
  // 默认留 pyin 是有实测依据的：四首人声对比中 rmvpe 在 245~291Hz 区间系统性偏高 5~6Hz，
  // 在 200Hz 分界附近可能翻转「男声/女声」判断；耗时两者持平（差 0.3~0.7s）。
  f0Engine: 'pyin',
  // 估调最多取多少秒（原来的行为是 40 秒）
  f0MaxSeconds: 40,
  // 按歌名记住起唱点的规则文件，相对 <singDir>/
  startOffsetsFile: 'start-offsets.json',
  // 仅用于结果提示文案，改成你机器人框架里发语音的工具名
  voiceToolName: 'send_voice',
  // 下面三项留空则由程序自动探测
  pythonPath: '',
  ffmpegPath: '',
  ytdlpPath: '',
  // 同时跑几首（默认 1：RVC 很吃显存，两个群同时点歌会 OOM）
  maxConcurrent: 1,
  // 中间产物保留几个 work 目录（0 = 不清理）
  keepWork: 3,
  // 成品目录保留几个 sing-*.mp3（0 = 不清理）
  keepOutput: 20,
};

function readConfigFile() {
  const file = process.env.SING_CONFIG || path.join(ROOT, 'config.json');
  try {
    let text = fs.readFileSync(file, 'utf8');
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function pick(name, envName, raw) {
  const fromEnv = process.env[envName];
  if (fromEnv !== undefined && fromEnv !== '') return fromEnv;
  if (raw[name] !== undefined && raw[name] !== '') return raw[name];
  return DEFAULTS[name];
}

function abs(p) {
  return path.isAbsolute(p) ? p : path.join(ROOT, p);
}

/** 相对某个目录解析；但允许直接给绝对路径。
 *  [坑] 早先用 path.join(singDir, x) 处理，用户在 SING_START_OFFSETS / SING_MODEL
 *  里写了绝对路径时会被拼成 "<singDir>\\E:\\cfg\\x.json" 这种永不存在的位置，
 *  表现为「规则文件明明写了却不生效」——静默失效，最难查。 */
function absUnder(dir, p) {
  const text = String(p ?? '');
  return path.isAbsolute(text) ? text : path.join(dir, text);
}

/** 可执行文件探测：给了就用，没给就试几个常见位置，最后交给 PATH。 */
function firstExisting(candidates, fallback) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return fallback;
}

export function loadConfig() {
  const raw = readConfigFile();

  const singDir = abs(String(pick('singDir', 'SING_DIR', raw)));
  const outputDir = abs(String(pick('outputDir', 'SING_OUTPUT_DIR', raw)));
  const isWin = process.platform === 'win32';

  const venvPython = isWin
    ? path.join(singDir, 'venv', 'Scripts', 'python.exe')
    : path.join(singDir, 'venv', 'bin', 'python');
  const pythonPath = firstExisting(
    [pick('pythonPath', 'SING_PYTHON', raw)].map((p) => (p ? abs(String(p)) : '')),
    venvPython,
  );

  const pythonDir = path.dirname(pythonPath);
  const ytdlpPath = firstExisting(
    [
      pick('ytdlpPath', 'SING_YTDLP', raw) ? abs(String(pick('ytdlpPath', 'SING_YTDLP', raw))) : '',
      path.join(pythonDir, isWin ? 'yt-dlp.exe' : 'yt-dlp'),
      path.join(singDir, 'venv', 'Scripts', 'yt-dlp.exe'),
      path.join(singDir, 'venv', 'bin', 'yt-dlp'),
    ],
    isWin ? 'yt-dlp.exe' : 'yt-dlp',
  );

  const ffmpegPath = firstExisting(
    [
      pick('ffmpegPath', 'SING_FFMPEG', raw) ? abs(String(pick('ffmpegPath', 'SING_FFMPEG', raw))) : '',
      path.join(pythonDir, isWin ? 'ffmpeg.exe' : 'ffmpeg'),
      path.join(singDir, 'bin', isWin ? 'ffmpeg.exe' : 'ffmpeg'),
    ],
    isWin ? 'ffmpeg.exe' : 'ffmpeg',
  );

  const num = (name, envName) => {
    const v = Number(pick(name, envName, raw));
    return Number.isFinite(v) && v > 0 ? v : DEFAULTS[name];
  };

  // 正整数/零：0 表示「不清理」
  const count = (name, envName) => {
    const v = Math.trunc(Number(pick(name, envName, raw)));
    return Number.isFinite(v) && v >= 0 ? v : DEFAULTS[name];
  };

  const modelName = String(pick('model', 'SING_MODEL', raw));

  return {
    singDir,
    outputDir,
    pythonPath,
    ytdlpPath,
    ffmpegPath,
    model: modelName,
    device: String(pick('device', 'SING_DEVICE', raw)),
    maxSeconds: Math.trunc(num('maxSeconds', 'SING_MAX_SECONDS')),
    defaultSeconds: Math.trunc(num('defaultSeconds', 'SING_DEFAULT_SECONDS')),
    targetF0: num('targetF0', 'SING_TARGET_F0'),
    femaleF0Threshold: num('femaleF0Threshold', 'SING_FEMALE_F0'),
    maxPitch: Math.trunc(num('maxPitch', 'SING_MAX_PITCH')),
    f0Engine: String(pick('f0Engine', 'SING_F0_ENGINE', raw)).toLowerCase(),
    f0MaxSeconds: num('f0MaxSeconds', 'SING_F0_MAX_SECONDS'),
    startOffsetsFile: absUnder(singDir, pick('startOffsetsFile', 'SING_START_OFFSETS', raw)),
    voiceToolName: String(pick('voiceToolName', 'SING_VOICE_TOOL', raw)),
    // 并发与磁盘：7x24 跑在机器人里时，这两项决定「会不会把显存和磁盘跑爆」
    maxConcurrent: Math.max(1, count('maxConcurrent', 'SING_MAX_CONCURRENT')),
    keepWork: count('keepWork', 'SING_KEEP_WORK'),
    keepOutput: count('keepOutput', 'SING_KEEP_OUTPUT'),
    workDir: path.join(singDir, 'work'),
    pthModel: absUnder(path.join(singDir, 'model'), modelName),
    coverScript: path.join(singDir, 'cover.py'),
    f0Script: path.join(singDir, 'est_f0.py'),
    rvcDir: path.join(singDir, 'rvc_cli'),
    rvcRuntimeDir: path.join(singDir, 'rvc'),
    contentvecCandidates: [
      path.join(singDir, 'rvc', 'models', 'embedders', 'contentvec', 'pytorch_model.bin'),
      path.join(singDir, 'contentvec', 'pytorch_model.bin'),
    ],
    rmvpeModel: path.join(singDir, 'model', 'rmvpe.pt'),
  };
}
