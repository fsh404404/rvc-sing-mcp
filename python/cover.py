# RVC 推理驱动：把一段人声换成目标音色。
#
# 用法： python cover.py <输入音频> <输出wav> [pitch] [f0_method]
# 音色模型：环境变量 SING_MODEL 指定文件名（相对 model/），不指定则自动挑 model/ 下的 .pth。
#
# 参数为什么这么定（都是实测结论，改动前请先测）：
#   volume_envelope=0.25  比 1 稳：高音不再被源音量包络顶到爆音
#   hop_length=128        非标准值会引入固定延迟（hop64 ≈ +32ms，咬字对不上）
#   index_rate=0.75       仅当存在 faiss 索引时启用，音色相似度更高
import glob
import os
import shutil
import sys
import traceback

# 目录按自身位置定位：整个目录搬到别的机器也能跑。
SING = os.path.dirname(os.path.abspath(__file__))
os.chdir(SING)                 # rvc_cli 用 cwd 解析模型路径
sys.path.insert(0, os.path.join(SING, "rvc_cli"))

# rmvpe 期望在 <cwd>/rvc_cli/rvc/models/predictors/rmvpe.pt
rmvpe_target = os.path.join(SING, "rvc_cli", "rvc", "models", "predictors", "rmvpe.pt")
rmvpe_src = os.path.join(SING, "model", "rmvpe.pt")
if os.path.isdir(os.path.dirname(rmvpe_target)):
    if not os.path.exists(rmvpe_target) and os.path.exists(rmvpe_src):
        shutil.copyfile(rmvpe_src, rmvpe_target)

# contentvec 是换音色必需的特征提取器。缺了它 RVC 会自己去 HuggingFace 下载，
# 网络不通时只会甩一个 URLError，很难看懂。这里提前拦住并说清楚。
# 两个常见位置都认：显式放在 rvc/models/embedders/ 下，或单独放 sing/contentvec/ 再链过去。
CONTENTVEC_CANDIDATES = [
    os.path.join(SING, "rvc", "models", "embedders", "contentvec", "pytorch_model.bin"),
    os.path.join(SING, "contentvec", "pytorch_model.bin"),
]
if not any(os.path.exists(p) for p in CONTENTVEC_CANDIDATES):
    raise SystemExit(
        "缺少 contentvec 特征模型（换音色必需）。放到下面任一位置：\n"
        "  %s\n  %s\n"
        "  或设置 HF_ENDPOINT=https://hf-mirror.com 后联网运行一次让它自动下载。"
        % (CONTENTVEC_CANDIDATES[0], CONTENTVEC_CANDIDATES[1])
    )

from rvc_cli.rvc.infer.infer import VoiceConverter  # noqa: E402


def pick_model():
    name = os.environ.get("SING_MODEL", "").strip()
    model_dir = os.path.join(SING, "model")
    if name:
        p = name if os.path.isabs(name) else os.path.join(model_dir, name)
        if not os.path.exists(p):
            raise SystemExit("找不到音色模型：%s" % p)
        return p
    found = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not found:
        raise SystemExit("model/ 下没有 .pth 音色模型（可用 convert_model.py 从 safetensors 转换）")
    return found[0]


def find_index():
    """自动发现 model/ 下的 faiss 检索索引（*.index）。

    有索引时 index_rate 从 0 提到 0.75，音色相似度明显更高；没有就纯模型推理。
    faiss 的 C++ 层读不了非 ASCII 路径：命中就先按内容哈希复制到 %TEMP%/rvc-index/ 再用。
    """
    model_dir = os.path.join(SING, "model")
    try:
        candidates = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.lower().endswith(".index")]
    except OSError:
        return ""
    if not candidates:
        return ""
    # 多个索引时优先与音色模型同名的
    stem = os.path.splitext(os.path.basename(pick_model()))[0].lower()
    candidates.sort(key=lambda p: (os.path.basename(p).lower().startswith(stem) is False, p))
    src = candidates[0]
    if all(ord(ch) < 128 for ch in src):
        return src
    import hashlib
    import tempfile
    tag = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
    cache = os.path.join(tempfile.gettempdir(), "rvc-index")
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, "voice-%s.index" % tag)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        shutil.copyfile(src, dst)
    return dst


def main():
    src = os.path.abspath(sys.argv[1])
    dst = os.path.abspath(sys.argv[2])
    pitch = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    f0_method = sys.argv[4] if len(sys.argv) > 4 else "rmvpe"
    model = pick_model()
    index_path = find_index()

    conv = VoiceConverter()
    print("[1/3] loading hubert ...", flush=True)
    conv.load_hubert("contentvec")
    print("[2/3] model: %s | index: %s" % (
        os.path.basename(model), os.path.basename(index_path) if index_path else "(无)"), flush=True)
    print("[3/3] converting ...", flush=True)
    conv.convert_audio(
        audio_input_path=src,
        audio_output_path=dst,
        model_path=model,
        index_path=index_path,
        pitch=pitch,
        f0_method=f0_method,
        index_rate=0.75 if index_path else 0.0,
        volume_envelope=0.25,
        protect=0.5,
        hop_length=128,
        split_audio=False,
        f0_autotune=False,
        embedder_model="contentvec",
        clean_audio=False,
        export_format="WAV",
        sid=0,
    )
    print("OK ->", dst, os.path.getsize(dst), "bytes", flush=True)


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
