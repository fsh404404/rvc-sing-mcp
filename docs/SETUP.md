# 安装细节

按顺序做，每步都有验收方式。所有命令在仓库根目录执行，Windows 用 PowerShell/CMD 均可。

## 0. Node 依赖

```bash
npm install
node --check src/mcp-sing.js   # 语法自检
```

## 1. Python 环境

```bash
python -m venv sing/venv
sing\venv\Scripts\python -m pip install -U pip
sing\venv\Scripts\python -m pip install -r requirements.txt
```

**验收**：`sing\venv\Scripts\python -c "import torch, demucs; print(torch.__version__)"`

### 没有 NVIDIA 显卡

`requirements.txt` 默认带 `+cu128`。改法：

1. 把 `--extra-index-url https://download.pytorch.org/whl/cu128` 删掉；
2. `torch` / `torchaudio` 去掉 `+cu128`；
3. `config.json` 里 `"device": "cpu"`。

CPU 下翻唱一首 30 秒片段大约几分钟，`mode=original` 不受影响。

### 国内网络慢

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

torch 的 cu128 轮子建议用官方 index 或镜像站，别用普通 PyPI 镜像（可能没有 cu 版本）。

## 2. yt-dlp 与 ffmpeg

两者都要能被找到。查找顺序：配置项 → venv 目录 → PATH。

```bash
sing\venv\Scripts\python -m pip install yt-dlp
# ffmpeg：任选一种
#   a) 装到 PATH
#   b) 在 config.json 里指定 "ffmpegPath": "D:/tools/ffmpeg/bin/ffmpeg.exe"
```

**验收**：`ffmpeg -version` 有输出；`node scripts/probe.mjs` 里 `ytdlp.ok` 与 `ffmpeg.ok` 都是 true。

## 3. RVC 推理代码（第三方）

`python/cover.py` 会 `sys.path.insert(sing/rvc_cli)` 再 `import rvc_cli`。拿到这份代码的两种方式：

**方式 A：从 wheel 解包**

```bash
sing\venv\Scripts\python -m pip download rvc-cli==1.8.0 --no-deps -d rvc_wheel
# 用解压工具打开 rvc_wheel/rvc_cli-1.8.0-py3-none-any.whl
# 把里面的 rvc/ 与 rvc_cli/ 两个目录复制到 sing/rvc_cli/
```

**方式 B：从已装好的机器复制**

任意一台跑通过 RVC 的机器，把它的 `rvc_cli` 目录整个拷过来即可（MIT 许可）。

**验收**：目录形如

```
sing/rvc_cli/
├─ rvc/            # 模型结构、推理 pipeline、predictors
└─ rvc_cli/        # 入口（VoiceConverter 等）
```

## 4. 模型

```
sing/
├─ model/
│  ├─ xiaoxiao.pth     # 音色模型（RVC v2，带 f0），文件名与 config.model 对应
│  ├─ rmvpe.pt         # 音高提取模型，首次运行会自动复制进 rvc_cli
│  └─ xxx.index        # 可选：faiss 检索索引，存在即自动启用 index_rate=0.75
├─ rvc/
│  └─ models/
│     ├─ embedders/contentvec/{config.json,pytorch_model.bin}   # 换音色必需
│     └─ predictors/rmvpe.pt
└─ contentvec/          # 也可以把 contentvec 放这里，再链到上面的位置
```

`contentvec` 是**换音色必需**的特征提取器（约 361 MB）。缺它时 RVC 会尝试联网下载，
网络不通会报 `urlopen error`，所以 `cover.py` 会提前拦下并提示这两条路径。

### safetensors 转 .pth

```bash
sing\venv\Scripts\python python\convert_model.py sing\model\xxx.safetensors sing\model\xxx.pth
```

转换需要同名的 `.json`（RVC 导出模型时配套生成）。

### 如果 contentvec 真的没有

```bash
set HF_ENDPOINT=https://hf-mirror.com
sing\venv\Scripts\python -c "import torch; from rvc_cli.rvc.infer.infer import VoiceConverter; VoiceConverter().load_hubert('contentvec')"
```

或者直接 `pip install huggingface_hub` 后手动下载 `lengyue233/content-vec-best` 的权重放进对应目录。

## 5. 配置

```bash
copy config.example.json config.json
```

最少要改的通常是 `model`（你的音色文件名）与 `voiceToolName`（你框架里的发语音工具名）。路径类字段留空即自动探测。

**验收**：

```bash
node scripts/probe.mjs                 # 全部项 ok:true
node scripts/probe.mjs --mode=ai       # 只看「能不能翻唱」（aiReady）
node scripts/probe.mjs --mode=original # 只看「能不能出原唱片段」
```

`torch` 一行会打印版本与 `is_available()`。`aiMissing` 列出的就是还差的那几样。

## 6. GPU 是否真的在工作

```bash
sing\venv\Scripts\python python\check-cuda.py
```

正常输出 `OK : torch 支持 sm_XXX` 且 `matmul` 平均 < 50 ms。若看到「不支持，会静默退回 CPU」，见 [TROUBLESHOOTING](TROUBLESHOOTING.md#新显卡上能出声但极慢)。

## 7. 接入 MCP 客户端

用 `examples/mcp.json.example` 的片段，把 `args` 改成你机器上的绝对路径。注册后**重启客户端**（MCP 工具列表是启动时加载的）。

第一次对话可以让 agent 先调用 `sing_probe`，确认无误再说「唱一首《晴天》」。

### 已经有跑通的机器：复用而不是重装

换机器、或把它接进已经在跑的机器人时，`sing/` 目录（venv + 模型 + `rvc_cli`，约 12 GB）**不需要重装**，指过去就行：

```
SING_DIR=<原机器上的 sing 目录>
SING_OUTPUT_DIR=<机器人框架的语音库目录>
SING_FFMPEG=<ffmpeg 可执行文件>
SING_VOICE_TOOL=<框架里发语音的工具名>
```

四个必查点与「现象 → 原因」对照表见 [README 的「接进已有机器人框架」](../README.md#接进已有机器人框架四条前提)。

### 客户端不透传环境变量

少数 MCP 客户端只把 `command` / `args` 传下去，不继承父进程环境。症状是 `env` 里怎么写都不生效。

最可靠的绕法是**不用环境变量**：在仓库根目录放一份 `config.json`（从 `config.example.json` 复制），程序默认就会读它，不依赖任何 env。

排查时先跑 `node scripts/probe.mjs`：它会把最终生效的路径打印出来，和你在客户端里配的对一下，就知道有没有透传成功。

## 8. 目录与磁盘占用参考

| 目录 | 说明 | 参考体积 |
|---|---|---|
| `sing/venv/` | Python 依赖 | ~6 GB |
| `sing/rvc_cli/` | 第三方推理代码 | ~173 MB |
| `sing/rvc/` | contentvec 等运行时模型资源 | ~361 MB |
| `sing/model/` | 你的音色模型 | 50~200 MB |
| `sing/work/` | 中间产物，可随时删 | 每首 50~150 MB |
| `output/` | 成品 mp3 | 每首 1~3 MB |
