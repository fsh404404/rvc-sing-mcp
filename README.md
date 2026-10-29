# rvc-sing-mcp

**给 AI Agent 一个「会唱歌」的 MCP 工具**：说一句「唱一首《晴天》」，它自己上网找歌、截片段、分离人声、用你的 RVC 音色模型翻唱、混回伴奏，最后把成品音频放到约定的目录里。

An MCP server that turns "sing me this song" into a ready-to-send audio file, using demucs + RVC voice conversion.

```
搜歌(B站/YouTube) → 下载音轨 → 截片段 → demucs 分离 → RVC 换音色 → 混音 → output/*.mp3
```

这个项目**只负责造文件**，不负责发送。发语音这一步交给你的机器人框架（QQ / Telegram / Discord / 微信…），因此可以接任何支持 MCP 的 agent。

---

## 特性

| | 说明 |
|---|---|
| 三个工具 | `sing_search` 搜歌 / `sing_cover` 出成品 / `sing_probe` 环境自检 |
| 两种模式 | `original` 只截原唱片段（只要 yt-dlp + ffmpeg）；`ai` 完整翻唱（需要 GPU 可选） |
| 自动定调 | 先估原唱人声中位基频，再把音域搬到目标音色的舒适区，而不是机械升八度 |
| 起唱点记忆 | 按歌名记住「第几秒开唱、唱多长」，避免每次都把前奏唱进去 |
| 索引自动启用 | `model/` 下存在 faiss `.index` 就自动开启检索（音色更贴），没有就纯模型推理 |
| 并发排队 | 同时来两首歌会排队而不是并行抢显存（`maxConcurrent`，默认 1） |
| 磁盘自清 | `sing/work/` 与 `output/` 按数量保留最近 N 个，7×24 跑不会撑爆盘 |
| 零硬编码 | 所有路径走配置/环境变量，clone 到哪都能跑 |
| 结构化自检 | 缺依赖时一次列全（含 contentvec / rmvpe / 输出目录可写），绝不「悄悄地」把 ai 模式降级成原唱 |
| 工程坑已踩平 | 老版 ffmpeg 的 amix 电平问题、新显卡算子缺失、B站 412、faiss 读不了中文路径 |

## 目录结构

```
rvc-sing-mcp/
├─ src/
│  ├─ mcp-sing.js       MCP server：三个工具与整条链路
│  └─ config.js         配置加载（环境变量 > config.json > 默认值）
├─ python/
│  ├─ cover.py          RVC 推理驱动（按自身位置定位，可整目录搬迁）
│  ├─ est_f0.py         估原唱中位基频（决定要不要升调）
│  ├─ convert_model.py  safetensors 音色模型 → .pth
│  ├─ cover-timing.py   分阶段耗时诊断（排障用）
│  └─ check-cuda.py     torch 是否真的在用 GPU（算力支持比对 + 计时）
├─ scripts/probe.mjs    不用接客户端也能跑一次自检
├─ examples/            MCP 注册示例、起唱点规则示例
├─ docs/                安装细节、排障手册、设计说明
├─ config.example.json  复制成 config.json 后按需修改
├─ requirements.txt     主依赖（含 cu128 torch）
└─ requirements-lock.txt 一台跑通机器的完整 freeze
```

## 环境要求

| 项 | 要求 |
|---|---|
| Node.js | ≥ 20 |
| Python | 3.10 ~ 3.12（实测 3.12） |
| ffmpeg | 任意版本（脚本已兼容老版本混音差异） |
| yt-dlp | 可执行文件，或装进 venv |
| GPU | 可选。N 卡 4GB+ 翻唱约 10 秒一首；纯 CPU 能跑但慢，`mode=original` 完全不需要 |
| 磁盘 | venv + 依赖约 6 GB，音色模型另算 |

## 快速开始

```bash
git clone https://github.com/fsh404404/rvc-sing-mcp.git
cd rvc-sing-mcp
npm install
cp config.example.json config.json   # Windows: copy config.example.json config.json
```

**1. 建 Python 环境**

```bash
python -m venv sing/venv
sing\venv\Scripts\python -m pip install -U pip
sing\venv\Scripts\python -m pip install -r requirements.txt
sing\venv\Scripts\python -m pip install yt-dlp     # 若 PATH 里没有 yt-dlp
```

**2. 放 RVC 推理代码**

`cover.py` 从 `sing/rvc_cli/` 导入 `rvc_cli`（第三方代码，本仓库不附带）：

```bash
pip download rvc-cli==1.8.0 --no-deps -d /tmp/rvc
# 解压 wheel，把其中的 rvc/ 与 rvc_cli/ 放进 sing/rvc_cli/
```

或者直接从任意已装好 RVC 的机器复制这个目录（MIT 许可，约 173 MB，其中大头是 rmvpe 模型）。

**3. 放模型**

`sing/` 目录最终长这样：

```
sing/
├─ venv/         Python 环境
├─ rvc_cli/      第三方推理代码（下一步获取）
├─ rvc/          RVC 运行时模型资源：contentvec 等（约 279 MB）
├─ model/        你的音色模型 + rmvpe + 可选 index
└─ work/         中间产物（可随时删）
```

| 文件 | 位置 | 说明 |
|---|---|---|
| 音色模型 `*.pth` | `sing/model/` | RVC v2 带 f0。safetensors 用 `convert_model.py` 转换 |
| 音高提取 `rmvpe.pt` | `sing/model/` | 首次运行会自动复制到 `rvc_cli/rvc/models/predictors/` |
| `contentvec` | `sing/rvc/models/embedders/contentvec/` | 特征提取器，**换音色必需**；也可放 `sing/contentvec/` |

`contentvec` 缺失时，RVC 会自己尝试去 HuggingFace 下载；网络不通只会甩一个 `URLError`，
所以 `cover.py` 会**提前拦截并提示**，不会让你对着 502 猜半天。

想让音色更贴，可以再放一个同名 `.index`（faiss 索引）到 `sing/model/`，会自动以 `index_rate=0.75` 启用。

**4. 自检**

```bash
node scripts/probe.mjs
```

输出里每一项 `ok:false` 就是还缺什么。也可以在 MCP 客户端里调 `sing_probe`。

**5. 接到你的 agent**

```json
{
  "mcpServers": {
    "sing": {
      "command": "node",
      "args": ["/abs/path/rvc-sing-mcp/src/mcp-sing.js"]
    }
  }
}
```

然后对 agent 说「唱一首《晴天》」——它会先 `sing_probe`，再 `sing_search` 挑版本，最后 `sing_cover(mode="ai")`。

### 接进已有机器人框架：四条前提

这套服务只造文件，接进你现成的机器人（QQ / Telegram / Discord…）时有四件事必须对上。对不上时的典型表现是「翻唱跑成功了，但一点声音都没有」。

**1. `node_modules` 不在仓库里**

`.gitignore` 排除了它。clone 后不装依赖就注册 MCP，客户端会直接启动失败：

```
Error: Cannot find module '@modelcontextprotocol/sdk/...'
```

**2. `sing/` 环境不进仓库**

venv、模型、`rvc_cli` 加起来约 12 GB，都在 `.gitignore` 里。**已经有跑通环境的机器不用重装**，把 `SING_DIR` 指过去复用即可：

```
SING_DIR=E:\your\existing\sing
```

重装一份的代价不小——光 torch 的 cu 轮子就 3 GB 起。

**3. 它不读你框架的配置文件**

本服务只认三样：环境变量 > 自己的 `config.json` > 内置默认值。宿主框架里的 `voice.dir`、`ffmpeg` 路径之类**完全不会被继承**，必须在注册时用 `env` 显式传进去：

```json
{
  "mcpServers": {
    "sing": {
      "command": "node",
      "args": ["<绝对路径>/rvc-sing-mcp/src/mcp-sing.js"],
      "env": {
        "SING_DIR": "<sing 目录>",
        "SING_OUTPUT_DIR": "<框架的语音库目录>",
        "SING_FFMPEG": "<ffmpeg 可执行文件>",
        "SING_VOICE_TOOL": "<你框架里发语音的工具名>"
      }
    }
  }
}
```

> 部分 MCP 客户端默认不透传父进程的环境变量。改了 `env` 却毫无反应时，改用 `config.json`（放仓库根目录，或用 `SING_CONFIG` 指到别处）。`scripts/probe.mjs` 就是踩了这个坑之后补出来的。

**4. 成品目录必须落在框架的语音库**

`SING_OUTPUT_DIR` 默认是仓库下的 `output/`。多数机器人在发语音时只认一个白名单目录（语音库），产物放错地方会被「文件不存在 / 不在允许列表」拒掉。**把 `SING_OUTPUT_DIR` 指到那个目录**即可；文件名本身已带 `sing-` 前缀，方便下游识别和限流。

排障对照：

| 现象 | 真实原因 |
|---|---|
| MCP 启动即失败，报 `Cannot find module` | 没执行 `npm install` |
| `sing_probe` 里 `venv` / `torch` 全是 false | `SING_DIR` 没指到实际环境 |
| 工具返回成功，但目录里没有文件 | `SING_OUTPUT_DIR` 指错或不可写 |
| 文件生成了，机器人却发不出去 | 产物不在框架的语音库 / 白名单内 |
| 改了 `env` 但参数不生效 | 客户端没传环境变量，改用 `config.json` |

## 工具

### `sing_probe`

无参数。返回每个环节的 `ok/detail/path`：`ytdlp` `ffmpeg` `venv` `torch` `demucs` `rvc` `coverScript` `f0Script` `voiceModel` `contentvec` `rmvpe` `rvcRuntime` `index` `outputDir` `startOffsetsFile`。

另外返回三个汇总字段，省得 agent 自己拼判断：

- `aiReady` / `aiMissing`：能不能真翻唱、还缺哪几样（与 `sing_cover(mode=ai)` 的前置检查同一份清单）；
- `originalReady`：能不能出原唱片段（只依赖 yt-dlp + ffmpeg + 可写的输出目录）。

`ytdlp` / `ffmpeg` 是**真的 spawn 一次**判的，不是 `existsSync`——装在 PATH 里也算就绪。

### `sing_search`

| 参数 | 类型 | 说明 |
|---|---|---|
| `query` | string | 歌名，可带歌手 |
| `limit` | int 1~8 | 返回几条，默认 5 |

先走 B 站公开搜索接口（带 buvid cookie，命中 412 会重取一次重试），失败再退 yt-dlp 的 `bilisearch` / `ytsearch`。

### `sing_cover`

| 参数 | 类型 | 说明 |
|---|---|---|
| `song` | string | 歌名，可带歌手 |
| `seconds` | int 5~maxSeconds | 截多长，默认 `defaultSeconds` |
| `startSeconds` | number | 从第几秒开始截 |
| `mode` | `original` \| `ai` | 默认 original |
| `candidate` | int 1~8 | 取搜索结果的第几个 |
| `pitch` | int -24~24 | 变调。**建议留空**，让脚本按音域自动定 |

返回 JSON：`name` `path` `title` `source` `url` `requestedSeconds` `startSeconds` `startOffsetRule` `mode` `pitch` `sourceMedianF0` `pitchNote` `bytes` `next`。

`next` 里提示的发送工具名由 `voiceToolName` 决定，改成你框架里真实的工具名即可。

## 配置

优先级：**环境变量 > `config.json` > 内置默认值**。所有相对路径相对仓库根目录。

| 键 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `singDir` | `SING_DIR` | `sing` | 工作目录：venv / 模型 / 脚本 / 中间产物 |
| `outputDir` | `SING_OUTPUT_DIR` | `output` | 成品目录 |
| `model` | `SING_MODEL` | `xiaoxiao.pth` | 音色模型文件名（相对 `sing/model/`，也可直接给绝对路径） |
| `device` | `SING_DEVICE` | `cuda` | demucs 设备：`cuda` / `cpu` |
| `maxSeconds` | `SING_MAX_SECONDS` | `60` | 单次最长秒数 |
| `defaultSeconds` | `SING_DEFAULT_SECONDS` | `30` | 不传 seconds 时的长度 |
| `targetF0` | `SING_TARGET_F0` | `250` | 定调目标基频（Hz） |
| `femaleF0Threshold` | `SING_FEMALE_F0` | `200` | 高于此值视为女声区，不再升调 |
| `maxPitch` | `SING_MAX_PITCH` | `12` | 自动定调上限（半音） |
| `f0Engine` | `SING_F0_ENGINE` | `auto` | 估调引擎：`auto`（有 rmvpe 就用，否则 pyin）/ `rmvpe` / `pyin` |
| `f0MaxSeconds` | `SING_F0_MAX_SECONDS` | `40` | 估调最多取多少秒人声 |
| `startOffsetsFile` | `SING_START_OFFSETS` | `start-offsets.json` | 起唱点规则文件（相对 `singDir`，也可给绝对路径） |
| `voiceToolName` | `SING_VOICE_TOOL` | `send_voice` | 仅用于提示文案 |
| `maxConcurrent` | `SING_MAX_CONCURRENT` | `1` | 同时跑几首。**别调大**：RVC 很吃显存，两个群同时点歌就是 OOM |
| `keepWork` | `SING_KEEP_WORK` | `3` | `sing/work/` 保留最近几个任务目录，`0`=不清理 |
| `keepOutput` | `SING_KEEP_OUTPUT` | `20` | `output/` 保留最近几个 `sing-*.mp3`，`0`=不清理 |
| `pythonPath` | `SING_PYTHON` | `sing/venv/...` | 自动探测 |
| `ffmpegPath` | `SING_FFMPEG` | PATH | 自动探测 |
| `ytdlpPath` | `SING_YTDLP` | venv/PATH | 自动探测 |

起唱点规则示例（`examples/start-offsets.example.json`）：

```json
{
  "晴天": { "start": 12, "seconds": 30 },
  "海阔天空": 20
}
```

键是歌名子串（不区分大小写），命中即覆盖估算值；以 `_` 开头的键会被忽略（可用来写注释）。

## 接手 7×24 跑之前的两件事

接进机器人之后它就不再是「手动开一次」的工具了，这两条不做，跑一周就出事：

1. **并发**：`maxConcurrent` 默认 `1`。请求来第二个时会**排队**，不会并行抢显存。
   排队发生在 MCP 进程内，客户端那边只是这次调用慢一点，不会报错。
2. **磁盘**：每次 `sing_cover` 会在 `sing/work/` 下留一个 50~150 MB 的目录（下载源 + 分离后的 wav）。
   默认只保留最近 `keepWork=3` 个任务目录、`output/` 只保留最近 `keepOutput=20` 个成品，
   清理在每次调用的收尾统一做（成功失败都清）。想自己管就设 `0` 关掉。

长任务还有进度通知：客户端传了 `progressToken` 时，`sing_cover` 会按
「搜歌 → 下载 → 截取 → 分离 → 转换 → 混音」六步上报 `notifications/progress`。
CPU 模式下翻唱要几分钟，没有这个很容易被客户端判成卡死。

## 为什么参数是这些值

这些不是拍脑袋定的，改动前建议先自己测一遍：

- **`hop_length=128`**：RVC 的非标准 hop 会给输出引入固定延迟。实测 `hop64` 相对源 +31.8 ms（咬字明显对不上），`hop128` −1.6 ms，`hop160` −1.7 ms。
- **`volume_envelope=0.25`**：比默认 1 稳，高音不再被源音量包络顶到爆音（高音段能量差从 −1.8 dB 收到 −0.7 dB）。
- **目标音域法（`targetF0=250`）**：以一首源中位 183 Hz 的歌为例，机械 +12 会顶到假声、丢帧率最高；把中位搬到 ~247 Hz（+5）丢帧与能量都更自然。
- **混音 `amix:normalize=0`**：`amix` 默认会把总和除以输入路数（−6 dB），成品比源低约 10 dB，前奏听起来像「卡住没声」。新 ffmpeg 用 `normalize=0`，老版本不认该参数时回退到 `volume=2` 补回那 6 dB。

## 定调引擎：为什么默认仍是 pyin

`est_f0.py` 支持两套引擎，默认 `pyin`（与改动前完全一致）：

| 引擎 | 说明 |
|---|---|
| `pyin` | librosa.pyin，纯 CPU，不加载任何模型。**默认** |
| `rmvpe` | 复用 `cover.py` 的 rmvpe 权重，与转换阶段同一条 f0 曲线 |
| `auto` | 先试 rmvpe，探测不到入口或权重就退回 pyin |

本机实测（RTX 5060 Laptop，四段 50 秒人声、各取前 40 秒）：

| 素材 | pyin | rmvpe | 差 | 耗时 pyin / rmvpe |
|---|---|---|---|---|
| 男声 A | 165.7 Hz | 165.5 Hz | −0.2 | 5.3 / 4.9 s |
| 男声 B | 197.0 Hz | 196.5 Hz | −0.5 | 4.8 / 4.5 s |
| 女声 A | 245.4 Hz | 250.4 Hz | +5.0 | 5.6 / 5.3 s |
| 女声 B | 285.2 Hz | 291.3 Hz | +6.1 | 5.1 / 4.6 s |

结论：**耗时基本持平**（rmvpe 快 0.3~0.7 s，但要加载 181 MB 权重），而 rmvpe 在女声区间系统性偏高 5~6 Hz。`femaleF0Threshold` 默认 200 Hz，男声 B 的 197.0 / 196.5 就压在这条线附近。所以默认保持 pyin —— 不拿一个未定性的偏差去换 0.4 秒。

- 结果里带 `ENGINE=`，`sing_cover` 返回里是 `sourceF0Engine`；
- 想对比：`SING_F0_ENGINE=rmvpe` 跑一次、`=pyin` 再跑一次，只看 `sourceMedianF0` 与 `pitch`；
- 中位数落在 `femaleF0Threshold ±8 Hz` 时，`pitchNote` 会额外提示「换引擎可能翻转判断」，并写出本次判定的方向；
- rmvpe 的入口在不同 RVC 包里位置和类名都不一样（rvc-cli 1.8.0 是 `rvc_cli/rvc/lib/predictors/RMVPE.py`、类名 `RMVPE0Predictor`；并且必须把 `sing/rvc_cli` 放进 `sys.path` 才能 import 到）。脚本按候选表依次探测，失败原因会带进 `REASON=`。

## 已知问题

新显卡（RTX 50 系 / sm_120）上「能出声但极慢」、B 站 412、`Cannot find module ... loadNapCat.js`、模型路径含中文、显存不足等，见 **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**。
安装细节见 **[docs/SETUP.md](docs/SETUP.md)**，设计取舍见 **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**。

## 这个仓库不包含

- **音色模型**（版权与体积原因）——请使用你拥有权利的声音，或自建模型。
- **`rvc_cli` 第三方推理代码**（MIT，需自行安装/复制）。
- **任何示例音频**。`output/` 与 `sing/work/` 都在 `.gitignore` 里。

## 合规

- 翻唱涉及**音乐作品的著作权**与**声音的人格权**。请只处理你拥有授权的内容（自己演唱的干声、明确允许使用的伴奏），并遵守你所在平台/地区的规则。
- 用他人声音训练或转换模型可能侵犯人格权，本项目不为这类使用提供便利，也不承担由此产生的责任。
- 依赖组件：RVC（MIT）、Demucs（MIT）、yt-dlp（Unlicense）、ffmpeg（LGPL/GPL，按你的构建而定）。

## License

MIT，见 [LICENSE](LICENSE)。
