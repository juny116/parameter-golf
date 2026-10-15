"""
Re-apply pruning + quantization + compression on existing final_model.pt
without retraining. Tests multiple pruning % and reports sizes.
"""
import io
import os
import sys
import torch
import zlib

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

sys.path.insert(0, os.path.dirname(__file__))
from train_gpt_improved import mixed_quantize_int6

MODEL_PATH = "final_model.pt"
CODE_PATH = "train_gpt_improved.py"
TARGET = 16_000_000


def compress(sd):
    quant_result, quant_meta = mixed_quantize_int6(sd, {"mlp", "attn", "bigram", "trigram"})
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    raw = buf.getvalue()
    if _COMPRESSOR == "zstd":
        blob = zstandard.ZstdCompressor(level=22).compress(raw)
    else:
        blob = zlib.compress(raw, 9)
    return len(blob)


def prune(sd, pct):
    out = {}
    for name, param in sd.items():
        p = param.clone()
        if p.ndim == 2 and p.numel() > 65536:
            thr = torch.quantile(p.abs().float().flatten(), pct / 100.0)
            p.masked_fill_(p.abs() < thr, 0.0)
        out[name] = p
    return out


def main():
    sd = torch.load(MODEL_PATH, map_location="cpu")
    with open(CODE_PATH, "r") as f:
        code_bytes = len(f.read().encode("utf-8"))

    print(f"Code: {code_bytes:,} bytes | Target: {TARGET:,} | Max model: {TARGET - code_bytes:,}")
    print(f"Compressor: {_COMPRESSOR}\n")
    print(f"{'Prune%':>7} {'Model':>13} {'Total':>13} {'Margin':>10}")
    print("-" * 48)

    for pct in [0, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
        pruned = prune(sd, pct)
        model_sz = compress(pruned)
        total = model_sz + code_bytes
        margin = TARGET - total
        tag = "OK" if margin >= 0 else "OVER"
        print(f"{pct:>6}% {model_sz:>12,} {total:>12,} {margin:>+9,}  {tag}")

        # Save the first one that fits
        if margin >= 0 and not os.path.exists("final_model_fit.int8.ptz"):
            print(f"  ^ Saving as final_model_fit.int8.ptz (prune {pct}%)")
            pruned2 = prune(sd, pct)
            quant_result, quant_meta = mixed_quantize_int6(pruned2, {"mlp", "attn", "bigram", "trigram"})
            buf = io.BytesIO()
            torch.save({"w": quant_result, "m": quant_meta}, buf)
            raw = buf.getvalue()
            if _COMPRESSOR == "zstd":
                blob = zstandard.ZstdCompressor(level=22).compress(raw)
            else:
                blob = zlib.compress(raw, 9)
            with open("final_model_fit.int8.ptz", "wb") as f:
                f.write(blob)


if __name__ == "__main__":
    main()
