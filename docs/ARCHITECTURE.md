# 设计说明

## 为什么是一个「只造文件」的 MCP server

机器人要唱歌，真正难的是**音频链路**，不是发送。发送方式千差万别（QQ 要转 SILK、Telegram 有 50MB 限制、Discord 直接传附件），把发送塞进来会让这个项目被某个平台绑死。

所以边界划在「产出一个 mp3 路径」：

```
Agent ──MCP──> rvc-sing-mcp ──> output/xxx.mp3 ──> 你的框架发送
```

好处是任何支持 MCP 的 agent 都能接，发送侧想怎么改都不影响音频链路。

## 链路

```
sing_cover(mode=ai)
  │
  ├─ 1. 搜索          B站公开接口（带 buvid cookie，412 重试一次）
  │                   └ 失败 → yt-dlp bilisearch / ytsearch
  ├─ 2. 下载          yt-dlp -f bestaudio -x --audio-format mp3
  ├─ 3. 截片段        ffmpeg -ss/-t → 44.1k 立体声
  ├─ 4. 人声分离      demucs --two-stems=vocals -d cuda
  │                   ├ vocals.wav        （要换音色的干声）
  │                   └ no_vocals.wav    （伴奏，最后混回来）
  ├─ 5. 定调          est_f0.py → librosa.pyin 中位基频
  │                   └ ≥200Hz 归 0；否则按 log2(targetF0/hz) 折算半音
  ├─ 6. 换音色        cover.py → RVC（contentvec + rmvpe + 音色模型 + 可选 faiss 索引）
  └─ 7. 混音          ffmpeg amix(normalize=0 优先) → loudnorm -15 LUFS → alimiter
```

`mode=original` 只走 1→3，所以**只依赖 yt-dlp + ffmpeg**，无卡也能用。

## 关键取舍

### 1. 缺依赖就报错，绝不静默降级

翻唱需要 venv / torch / demucs / rvc_cli / cover.py / 音色模型六样。少任何一样，`sing_cover(mode=ai)` 会一次列全缺什么，而不是「悄悄给你发一段原唱」。

理由：静默降级的失败是**语义级**的——人以为听到了自己的音色，其实没有，很难事后发现。宁可失败也不要撒谎。

### 2. 定调按「音域」而不是「八度」

常见做法是男声歌统一 +12。问题是 +12 会把源人声顶到目标音色的假声区，丢帧率飙升、听感发尖。

这里的做法是先把源人声中位基频搬到 `targetF0`（默认 250 Hz）附近，再四舍五入到半音，并限制在 `maxPitch` 内。中位基频已经高于 `femaleF0Threshold` 时一律不升调。

实测（源 183 Hz）：`p0` 丢帧 6.0%、`p12` 丢帧 8.7%，而 `p5`（输出约 247 Hz）丢帧 4.7% —— 所以「搬到中心」比「搬八度」自然。

### 3. 起唱点规则外置成 JSON

前奏长度是**歌曲属性**，不是模型能可靠猜出来的东西。每次让模型估都会把念白/前奏唱进去。

`start-offsets.json` 把它变成可编辑的数据：键是歌名子串，命中就强制覆盖。规则文件坏了也只是退回请求参数，不影响出歌。

### 4. 混音要兼容老 ffmpeg

`amix` 默认 `normalize=1`，把结果除以输入路数（约 −6 dB）。新版本可以 `normalize=0` 关掉，但老构建（如 2018 年的 N-92722）不认这个参数会直接报错。

所以混音是**两级尝试**：先 `normalize=0`，失败回退 `volume=2` 补回 6 dB，两条都带 `loudnorm`（统一到 −15 LUFS）与 `alimiter`（防削顶）。

### 5. 配置解析顺序：环境变量 > config.json > 默认值

- 默认值让「clone 下来不改配置也能跑通自检」；
- `config.json` 存机器相关设置，但**不入库**（`.gitignore`）；
- 环境变量用于多实例/CI 覆盖，比如同时跑 `cuda` 和 `cpu` 两个 profile。

所有相对路径都相对**仓库根目录**解析，不含任何写死的盘符。

## 错误与观测

| 层面 | 做法 |
|---|---|
| 依赖缺失 | `sing_probe` 结构化返回 `ok/detail/path` |
| 子进程失败 | 捕获 stderr 末 3 行拼进错误信息 |
| 疑难怪慢 | `cover-timing.py` 给 Pipeline 的 `get_f0` / `voice_conversion` / `pipeline` 打点，打印设备与峰值显存 |
| GPU 假可用 | `check-cuda.py` 比对 `arch_list` 并计时，而不是只看 `is_available()` |
| MCP 进程存活 | 只捕获 `uncaughtException` / `unhandledRejection` 记日志，不退出（退出会让客户端丢工具） |

## 性能参考

| 场景 | 硬件 | 30 秒片段耗时 |
|---|---|---|
| 搜索 + 下载 | 网络 50 Mbps | 5~15 s |
| demucs 分离 | RTX 5060 Laptop | 8~15 s |
| RVC 转换 | RTX 5060 Laptop | 5~10 s |
| 混音 + 编码 | 任意 | 2~3 s |
| **合计** | | **约 30~45 s** |

CPU 模式（`device=cpu`）下 demucs 与 RVC 会慢一个数量级，30 秒片段大约几分钟。

## 想扩展的话

| 想加什么 | 改哪里 |
|---|---|
| 新的音源（网易云/QQ音乐） | `searchSongs()` 里加一条分支，返回同样的 `{title,url,duration,source}` |
| 换分离模型（如 UVR/MDX） | `separateVocals()`，保持返回 `{vocals, instrumental}` |
| 后处理（混响、均衡） | `mixTracks()` 的 filter 链 |
| 多音色切换 | 每次调用传 `SING_MODEL` 到子进程环境（`run()` 支持 `env` 覆盖） |
| 批量产出 | 直接调 `sing_cover` 多次；`output/` 的文件名带秒级时间戳不会冲突 |
