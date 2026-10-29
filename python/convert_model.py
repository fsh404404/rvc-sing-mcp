"""把 RVC 的 safetensors 音色模型转成 cover.py 用的 .pth。

用法：
    python convert_model.py <输入.safetensors> [输出.pth]

同目录下需要有一个同名 .json（RVC 导出模型时自带），里面含 config / f0 / version / info。
不给输出路径时，写到同目录同名 .pth。
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file


def convert(src, dst=None):
    src = os.path.abspath(src)
    base, _ = os.path.splitext(src)
    cfg_path = base + ".json"
    if not os.path.exists(cfg_path):
        raise SystemExit("找不到配套配置：%s（RVC 导出时应同时生成 .json）" % cfg_path)
    dst = dst or (base + ".pth")

    sd = load_file(src)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    keys = list(sd.keys())
    print("参数量:", len(keys), "| 首个键:", keys[0])
    # 部分导出的权重带 "weight." 前缀，RVC 推理时期望去掉
    if keys[0].startswith("weight."):
        sd = {k[len("weight."):]: v for k, v in sd.items()}

    cpt = {
        "config": cfg["config"],
        "weight": sd,
        "f0": cfg.get("f0", 1),
        "version": cfg.get("version", "v2"),
        "info": cfg.get("info", ""),
    }
    torch.save(cpt, dst)
    print("已保存:", dst)
    print("采样率:", cfg["config"][-1], "| 说话人数:", sd["emb_g.weight"].shape[0])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
