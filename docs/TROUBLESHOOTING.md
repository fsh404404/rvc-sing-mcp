# 排障手册

先跑这两条，多数问题一眼可见：

```bash
node scripts/probe.mjs                             # 缺什么依赖
sing\venv\Scripts\python python\check-cuda.py      # GPU 到底在不在工作
```

---

## 新显卡上「能出声但极慢」

**现象**：5 秒音频要几百秒；`nvidia-smi` 里 GPU 利用率只有 5~11%；某个 CPU 核心跑满；显存占用正常；**不报错**。

**原因**：torch 的 CUDA 轮子没有该显卡的算子（RTX 50 系是 `sm_120`），算子被**静默退回 CPU**。`torch.cuda.is_available()` 仍然是 True，所以很容易误判成「已经在用 GPU」。

**诊断**：

```bash
sing\venv\Scripts\python python\check-cuda.py
# 看 arch_list 里有没有你的 capability
```

**修复**：装 cu128 及以上的 torch：

```bash
sing\venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu128 ^
  torch==2.7.1+cu128 torchaudio==2.7.1+cu128
```

`cover-timing.py` 可以进一步定位卡在哪一段：

```bash
sing\venv\Scripts\python python\cover-timing.py 输入.wav 输出.wav 0 rmvpe
```

---

## 成品「对不上词」/ 开头有固定延迟

**原因**：`hop_length` 不是标准值。RVC 在这种配置下会给输出引入固定偏移。

**实测参考**（同一段素材）：`hop64` = +31.8 ms、`hop128` = −1.6 ms、`hop160` = −1.7 ms。

**修复**：保持 `cover.py` 里的 `hop_length=128`。改了它就要重新核对同步。

---

## 前奏听着「卡住 / 没声音」、整体比原曲小

**原因**：ffmpeg 的 `amix` 默认 `normalize=1`，会把两路混合结果除以输入路数（≈ −6 dB），再叠加老 ffmpeg 不认 `normalize=0` 参数，最终成品比源低约 10 dB。

**修复**：`mcp-sing.js` 的 `mixTracks()` 已经做了两级回退（先试 `normalize=0`，失败回退 `volume=2` + `loudnorm` + `alimiter`）。如果仍然偏小，先更新 ffmpeg 到新版本，再检查 `[1:a]volume=0.72` 这个伴奏增益是否符合你的素材。

**自查**：

```bash
ffmpeg -hide_banner -loglevel error -i output/xxx.mp3 -filter_complex ebur128 -f null -
```

看 `I:` 是否在 −16 LUFS 附近。

---

## 搜歌失败 / B 站 412

- 脚本已经会先访问首页拿 `buvid3` cookie，并在 412 时重取一次。
- 仍然失败通常是风控加严：给 yt-dlp 配 cookies（`--cookies-from-browser chrome` 或 `cookies.txt`），或换 YouTube 源。
- `sing_search` 返回空但 `sing_cover` 能出歌：说明走的是 yt-dlp 兜底路径，正常。

---

## `ffmpeg` / `yt-dlp` 找不到

```
Error: spawn ffmpeg.exe ENOENT
```

按 `config.json` 里的 `ffmpegPath` / `ytdlpPath` 显式指定绝对路径，或用环境变量 `SING_FFMPEG` / `SING_YTDLP`。Windows 上路径含空格要保留引号（JSON 字符串本身不需要）。

> `sing_probe` 是真的 spawn 一次 `ffmpeg -version` 来判定，所以**装在 PATH 上也算就绪**。
> 如果它报 `ok:false` 且 detail 是「找不到可执行文件」，那是真的不在配置里也不在 PATH 上——
> 注意 MCP 客户端启动的子进程 PATH 可能和你终端里的不同（见上面「客户端不透传环境变量」）。

---

## demucs 第一次运行卡在下载

首次运行要拉 `htdemucs` 权重。网络不好会卡很久或失败。

- 脚本在检测到本地已有 `.th` 缓存时会自动设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，避免每次联网校验。
- 如果缓存不完整：临时 `unset` 这两个变量，用镜像拉一次：

```bash
set HF_ENDPOINT=https://hf-mirror.com
sing\venv\Scripts\python -m demucs -n htdemucs --two-stems=vocals -d cpu 测试.wav
```

---

## faiss 索引报错 / 索引不生效

- faiss 的 C++ 层读不了非 ASCII 路径。`cover.py` 检测到索引路径含中文时会按内容哈希复制到 `%TEMP%/rvc-index/` 再用，通常不用管。
- 索引与音色模型不配套时会「更不像」：索引必须由**同一个模型**训练产生。没有就删掉 `.index`，走纯模型推理。

---

## 显存不足（CUDA out of memory）

按代价从低到高：

1. 缩短 `seconds`（30 秒以内最稳）；
2. `config.json` 改 `"device": "cpu"`（慢但不会炸）；
3. 在 `cover.py` 里把 `split_audio` 改成 `True`（长音频分段，边界可能有轻微接缝）。

---

## 启动时 `Config file not found: rvc_cli\rvc\configs\v2\40000.json`

无害。RVC 会退回内置默认配置，只在音色模型自带 config 时才需要那些文件。

---

## MCP 客户端里看不到 `sing_*` 工具

1. 客户端注册后**重启**（工具列表启动时加载）；
2. `command` 用绝对路径的 node，`args` 指向 `src/mcp-sing.js` 的绝对路径；
3. 手动跑一次 `node src/mcp-sing.js`，正常表现是**不输出任何 stdout**、立即等待 stdin（stdout 是 MCP 协议通道，日志只能走 stderr）；
4. 若客户端有工具白名单/守卫，把 `mcp__sing__` 前缀加进去。

Windows 上如果 QQ 之类客户端装在含空格路径（`D:\Program Files\...`）并被注入式加载器启动，可能出现
`Cannot find module 'D:\Program ...\loadNapCat.js'` —— 这是加载器把命令行按空格切断，与本项目无关。
处理方式：把客户端装到无空格路径（如 `D:\QQNT`），或升级加载器。

---

## 出歌了但没人收到

本项目**不发消息**。`sing_cover` 只是把文件写进 `output/`，真正的发送由你的机器人框架完成。看返回 JSON 里的 `path` 与 `next` 字段，把文件名交给框架的发语音工具（工具名配到 `voiceToolName`）。

---

## 想复现某个人的问题，需要收集什么

```bash
node scripts/probe.mjs > probe.txt
sing\venv\Scripts\python python\check-cuda.py > cuda.txt
sing\venv\Scripts\python -m pip freeze > freeze.txt
ffmpeg -version >> cuda.txt
```

外加 `sing_cover` 返回的 JSON（`pitchNote`、`sourceMedianF0` 对定位定调问题很关键）。
