# 估算一段人声的中位基频（Hz），用于判断「原唱是男声还是女声」，避免乱升调。
#
# 用法：
#   python est_f0.py <音频> [--engine=auto|rmvpe|pyin] [--max-seconds=40]
#
# 输出（给 mcp-sing.js 解析）：
#   MEDIAN_F0=369.8 VOICED_RATIO=0.82 ENGINE=rmvpe SECONDS=40
#   MEDIAN_F0=0 REASON=librosa_missing ENGINE=none
#
# 为什么要有 engine 开关：
#   cover.py 换音色时用 rmvpe 提 f0。估调这里如果另用 librosa.pyin，
#   两套算法对同一段人声的中位数可能差一点点，个别歌会因此差一个半音；
#   pyin 在 CPU 上也要十几秒（hop 256、全曲 40 秒）。
#   用同一个 rmvpe 既快，又保证「估调的依据」和「转换的依据」是同一条曲线。
#
#   但 rmvpe 的入口在不同 RVC 包里位置不一样（rvc_cli.rvc.infer.lib / rvc_cli.rvc.lib /
#   infer.lib …），探测不到就自动退回 pyin，绝不因为引擎不可用让整条链路失败。
#   想固定用哪套就传 --engine=pyin 或 --engine=rmvpe。
import importlib
import os
import sys

SING = os.path.dirname(os.path.abspath(__file__))
PYIN_SR = 22050
RMVPE_SR = 16000

# 和 cover.py 一样：真正的包在 sing/rvc_cli/rvc_cli/ 里，
# 必须把 sing/rvc_cli 放进 sys.path，`import rvc_cli` 才会解析到那个包。
# 不加这一步，`rvc_cli` 会命中 sing/rvc_cli/ 这个同名目录（它下面没有 rvc.lib），
# 于是所有候选都报 "No module named 'rvc_cli.rvc.lib'"。
RVC_CLI_DIR = os.path.join(SING, "rvc_cli")
if os.path.isdir(RVC_CLI_DIR) and RVC_CLI_DIR not in sys.path:
    sys.path.insert(0, RVC_CLI_DIR)

# rmvpe 的入口在不同 RVC 包里位置、类名、大小写都不一样：
#   rvc-cli 1.8.0（实测）→ rvc_cli/rvc/lib/predictors/RMVPE.py，类名 RMVPE0Predictor
#   其它常见发行版    → rvc/lib/rmvpe.py 或 infer/lib/rmvpe.py，类名 RMVPE
# 从最可能到最不可能试一遍，全部失败会把这些候选名一起报出来。
RMVPE_ENTRY_CANDIDATES = [
    ("rvc_cli.rvc.lib.predictors.RMVPE", "RMVPE0Predictor"),
    ("rvc_cli.rvc.lib.predictors.rmvpe", "RMVPE"),
    ("rvc_cli.rvc.infer.lib.rmvpe", "RMVPE"),
    ("rvc_cli.rvc.lib.rmvpe", "RMVPE"),
    ("rvc_cli.rmvpe", "RMVPE"),
    ("infer.lib.rmvpe", "RMVPE"),
]
# rmvpe 权重：cover.py 会把它复制进 rvc_cli，两个位置都认
RMVPE_WEIGHTS = [
    os.path.join(SING, "rvc_cli", "rvc", "models", "predictors", "rmvpe.pt"),
    os.path.join(SING, "model", "rmvpe.pt"),
]


def parse_args(argv):
    opts = {"audio": None, "engine": "auto", "max_seconds": 40.0}
    for arg in argv[1:]:
        if arg.startswith("--engine="):
            opts["engine"] = arg.split("=", 1)[1].strip().lower()
        elif arg.startswith("--max-seconds="):
            try:
                value = float(arg.split("=", 1)[1])
                if value > 0:
                    opts["max_seconds"] = value
            except ValueError:
                pass
        elif not arg.startswith("--") and opts["audio"] is None:
            opts["audio"] = arg
    return opts


def load_audio(path, sr, max_seconds):
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    limit = int(sr * max_seconds)
    if y.size > limit:
        y = y[:limit]
    return y, sr


def median_of(values):
    import numpy as np

    if values is None or len(values) == 0:
        return 0.0
    return float(np.median(values))


def f0_pyin(y, sr):
    """librosa.pyin：不依赖 RVC，任何环境只要有 librosa 就能跑，但慢。"""
    import librosa
    import numpy as np

    f0, _voiced_flag, _probs = librosa.pyin(
        y, fmin=65, fmax=1200, sr=sr, hop_length=256, frame_length=2048, fill_na=np.nan,
    )
    vals = f0[np.isfinite(f0)]
    return vals, float(len(f0))


def import_rmvpe():
    """返回 (类, 模块名, 失败原因列表)。失败原因要带出来：
    「模块不存在」和「模块在但 import 报错」是两种完全不同的毛病，
    早先一律报「找不到入口」，把 torch 版本冲突之类的问题也伪装成路径问题。"""
    failures = []
    for module_name, class_name in RMVPE_ENTRY_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 —— 记录原因，继续试下一个
            failures.append("%s: %s" % (module_name, exc))
            continue
        klass = getattr(module, class_name, None)
        if klass is None:
            failures.append("%s: 没有 %s 这个类" % (module_name, class_name))
            continue
        return klass, module_name, failures
    return None, None, failures


def build_rmvpe(klass, weights, is_half, device):
    """不同 RVC 版本的构造签名不一样，按常见几种依次试，全失败再抛。"""
    attempts = [
        lambda: klass(weights, is_half, device),
        lambda: klass(weights, is_half),
        lambda: klass(weights, device),
        lambda: klass(weights),
        lambda: klass(model_path=weights, is_half=is_half, device=device),
    ]
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:  # 签名不匹配，换下一种
            last = exc
    raise RuntimeError("rmvpe 构造失败：%s" % last)


def preflight_rmvpe():
    """先做便宜的入口/权重检查：这两步失败与「librosa 没装」是两回事，
    混在一起上报会把人带偏（先看到 librosa 报错，会误以为装个 librosa 就好了）。"""
    klass, module_name, failures = import_rmvpe()
    if klass is None:
        detail = "; ".join(failures[:2]) if failures else "没有候选"
        raise RuntimeError("找不到 rmvpe 入口（%s）" % detail.replace("\n", " ")[:240])
    weights = next((p for p in RMVPE_WEIGHTS if os.path.exists(p)), None)
    if weights is None:
        raise RuntimeError("找不到 rmvpe.pt（放 sing/model/ 或 sing/rvc_cli/rvc/models/predictors/）")
    return klass, module_name, weights


def f0_rmvpe(audio, entry):
    """与 cover.py 转换阶段同一个 rmvpe，代价低且口径一致。"""
    klass, module_name, weights = entry

    import numpy as np
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = device == "cuda"
    model = build_rmvpe(klass, weights, is_half, device)

    # infer_from_audio 内部直接 torch.from_numpy，必须是连续 float32
    samples = np.ascontiguousarray(np.asarray(audio, dtype="float32"))
    if device == "cuda":
        with torch.no_grad():
            f0 = model.infer_from_audio(samples, thred=0.03)
    else:
        f0 = model.infer_from_audio(samples, thred=0.03)

    arr = np.asarray(f0, dtype="float32")
    vals = arr[np.isfinite(arr) & (arr > 0)]
    print("[est_f0] rmvpe via %s on %s" % (module_name, device), file=sys.stderr)
    return vals, float(arr.size)


def run_engine(name, max_seconds, audio_path):
    """返回 (vals, frames)；失败抛异常，由调用方决定回退还是放弃。"""
    if name == "rmvpe":
        entry = preflight_rmvpe()
        y, _sr = load_audio(audio_path, RMVPE_SR, max_seconds)
        if y.size == 0:
            return None, 0.0
        return f0_rmvpe(y, entry)
    y, sr = load_audio(audio_path, PYIN_SR, max_seconds)
    if y.size == 0:
        return None, 0.0
    return f0_pyin(y, sr)


def main():
    opts = parse_args(sys.argv)
    if not opts["audio"] or not os.path.exists(opts["audio"]):
        # 上层靠 MEDIAN_F0=0 判定「估不出来」，这里不能崩，否则 stderr 会盖掉 stdout 的解析
        print("MEDIAN_F0=0 VOICED_RATIO=0.00 ENGINE=none REASON=audio_missing", flush=True)
        return 0

    engine = opts["engine"] if opts["engine"] in ("rmvpe", "pyin") else "auto"
    order = ["rmvpe", "pyin"] if engine == "auto" else [engine]

    errors = []
    for name in order:
        try:
            vals, frames = run_engine(name, opts["max_seconds"], opts["audio"])
        except Exception as exc:
            errors.append("%s: %s" % (name, exc))
            print("[est_f0] %s 不可用：%s" % (name, exc), file=sys.stderr)
            continue
        if vals is None or len(vals) == 0:
            errors.append("%s: 没有有效帧" % name)
            continue
        voiced = float(len(vals)) / frames if frames else 0.0
        print(
            "MEDIAN_F0=%.1f VOICED_RATIO=%.2f ENGINE=%s SECONDS=%.1f"
            % (median_of(vals), voiced, name, opts["max_seconds"]),
            flush=True,
        )
        return 0

    print(
        "MEDIAN_F0=0 VOICED_RATIO=0.00 ENGINE=none REASON=%s"
        % ("; ".join(errors) or "unknown").replace("\n", " ")[:300],
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
