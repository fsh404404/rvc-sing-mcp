# 估算一段人声的中位基频（Hz），用于判断「原唱是男声还是女声」，避免乱升调。
#
# 用法： python est_f0.py <音频>
# 输出： MEDIAN_F0=369.8 VOICED_RATIO=0.82
import sys

import librosa
import numpy as np

SR = 22050
p = sys.argv[1]
y, _ = librosa.load(p, sr=SR, mono=True)
if len(y) > SR * 40:          # 只取前 40 秒，够估中位数了
    y = y[: SR * 40]
f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=1200, sr=SR, hop_length=256,
                            frame_length=2048, fill_na=np.nan)
vals = f0[np.isfinite(f0)]
if vals.size == 0:
    print("MEDIAN_F0=0 VOICED_RATIO=0")
else:
    print(f"MEDIAN_F0={float(np.median(vals)):.1f} VOICED_RATIO={vals.size / len(f0):.2f}")
