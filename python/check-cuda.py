# 验证 torch 能否在本机 GPU 上**真正**跑（不只是 is_available）。
#
# 真正的坑是「算力支持列表」：新显卡（如 50 系 Blackwell sm_120）用旧 cu 轮子时
# 没有对应算子，torch 会把算子**静默退回 CPU** —— is_available 仍是 True、
# matmul 也能跑，但推理会慢几十倍（5 秒音频要几百秒，GPU 利用率 5~11%）。
# 所以这里必须比对 arch_list 并计时，不能只看 is_available。
#
# 用法： python check-cuda.py        退出码 0=正常，2=算力不支持
import sys
import time

import torch

print("torch     :", torch.__version__)
print("cuda build:", torch.version.cuda)
print("available :", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("!! CUDA 不可用，RVC 无法上 GPU")
    sys.exit(1)

print("device    :", torch.cuda.get_device_name(0))
major, minor = torch.cuda.get_device_capability(0)
want = "sm_%d%d" % (major, minor)
print("capability:", want)

try:
    arch_list = list(torch.cuda.get_arch_list())
except Exception:
    arch_list = []
print("arch_list :", ", ".join(arch_list) or "(取不到)")

if arch_list and want not in arch_list:
    print("!! torch 没有编译 %s 的算子：会静默退回 CPU 跑（极慢，但不报错）" % want)
    print("   → 换 cu128 及以上的 torch/torchaudio，或把 device 改成 cpu 少等")
    supported = False
else:
    print("OK        : torch 支持 %s" % want)
    supported = True

torch.cuda.synchronize()
a = torch.randn(2048, 2048, device="cuda")
t0 = time.time()
for _ in range(10):
    a = a @ a
torch.cuda.synchronize()
elapsed = (time.time() - t0) / 10 * 1000
print("matmul    : 平均 %.1f ms/次" % elapsed)
if elapsed > 200:
    print("!! 这个耗时不像在 GPU 上跑（正常应 < 50ms）")

print("显存      : %.0f / %.0f MiB 已用" % (
    torch.cuda.memory_allocated() / 1024 / 1024,
    torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
))
sys.exit(0 if supported else 2)
