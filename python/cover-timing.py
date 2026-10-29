"""带分阶段计时的 cover.py：定位「显卡明明在，却全在 CPU 跑」到底卡在哪一段。

只在排障时用，不参与正式出歌。

用法： python cover-timing.py <输入音频> <输出wav> [pitch] [f0_method]

怎么看结果：
  - arch_list 里没有本卡算力，且 convert 阶段极慢  → torch 版本不支持该显卡（换 cu128+）
  - 各阶段都很快但成品对不上词                    → 看 hop_length（非 64/128 会引入固定延迟）
"""
import glob
import os
import sys
import time
import traceback

SING = os.path.dirname(os.path.abspath(__file__))
os.chdir(SING)
sys.path.insert(0, os.path.join(SING, "rvc_cli"))


def log(msg):
    print("[timing] " + msg, flush=True)


def pick_model():
    name = os.environ.get("SING_MODEL", "").strip()
    model_dir = os.path.join(SING, "model")
    if name:
        return name if os.path.isabs(name) else os.path.join(model_dir, name)
    preferred = [p for p in sorted(glob.glob(os.path.join(model_dir, "*.pth")))]
    return preferred[0] if preferred else os.path.join(model_dir, "voice.pth")


def main():
    src = os.path.abspath(sys.argv[1])
    dst = os.path.abspath(sys.argv[2])
    pitch = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    f0_method = sys.argv[4] if len(sys.argv) > 4 else "rmvpe"

    t_import = time.time()
    import torch
    from rvc_cli.rvc.infer.infer import VoiceConverter
    from rvc_cli.rvc.infer import pipeline as pipe_mod
    log("import rvc_cli + torch: %.2fs" % (time.time() - t_import))
    log("torch %s | cuda %s | device %s" % (
        torch.__version__, getattr(torch.version, "cuda", None),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"))
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = list(torch.cuda.get_arch_list())
        want = "sm_%d%d" % (major, minor)
        log("算力 %s | torch 支持: %s | %s" % (
            want, ", ".join(arch),
            "OK" if (not arch or want in arch) else "!! 不支持，算子会退回 CPU（慢几十倍）"))

    Pipeline = pipe_mod.Pipeline
    for name in ("get_f0", "voice_conversion", "pipeline"):
        orig = getattr(Pipeline, name, None)
        if orig is None:
            continue

        def make(orig_fn, label):
            def wrapper(self, *a, **k):
                t = time.time()
                r = orig_fn(self, *a, **k)
                extra = ""
                if torch.cuda.is_available():
                    extra = " | 峰值显存 %.0f MiB" % (torch.cuda.max_memory_allocated() / 1024 / 1024)
                log("%s: %.2fs | 设备 %s%s" % (label, time.time() - t, getattr(self, "device", "?"), extra))
                return r
            return wrapper
        setattr(Pipeline, name, make(orig, name))

    conv = VoiceConverter()

    t = time.time()
    conv.load_hubert("contentvec")
    log("load_hubert: %.2fs" % (time.time() - t))
    try:
        devs = {str(p.device) for p in conv.hubert_model.parameters()}
        log("hubert 参数所在设备: %s" % (", ".join(devs) or "无参数"))
    except Exception as exc:
        log("读 hubert 设备失败: %s" % exc)

    t_total = time.time()
    conv.convert_audio(
        audio_input_path=src,
        audio_output_path=dst,
        model_path=pick_model(),
        index_path="",
        pitch=pitch,
        f0_method=f0_method,
        index_rate=0.0,
        volume_envelope=0.25,   # 与正式 cover.py 保持一致，耗时才有可比性
        protect=0.5,
        hop_length=128,
        split_audio=False,
        f0_autotune=False,
        embedder_model="contentvec",
        clean_audio=False,
        export_format="WAV",
        sid=0,
    )
    log("convert_audio 总计: %.2fs" % (time.time() - t_total))
    log("DONE -> %s (%d bytes)" % (dst, os.path.getsize(dst)))


try:
    main()
except Exception:
    traceback.print_exc()
    sys.exit(1)
