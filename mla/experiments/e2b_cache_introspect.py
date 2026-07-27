"""E2b：MiniCPM3-4B 缓存取证——逻辑字节 vs 母张量占用（视图钉内存）。

prefill 1024 token 后直接数 past_key_values 里张量的 nbytes 与 backing storage，
解释 e2 差值法测到的 1,111,040 B/tok 的构成。输出 results/e2b_cache_introspect.json。
用 transformers 4.46 venv 运行。
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "/root/autodl-fs/models/MiniCPM3-4B"
L = 1024

tok = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
ids = tok("The history of computing is a story of layered abstractions. " * 100,
          return_tensors="pt").input_ids[:, :L].cuda()
with torch.inference_mode():
    pkv = model(input_ids=ids, use_cache=True).past_key_values

tensors = (list(pkv.key_cache) + list(pkv.value_cache)) if hasattr(pkv, "key_cache") \
    else [t for layer in pkv for t in layer]
logical = sum(t.nbytes for t in tensors)
storage = sum(t.untyped_storage().nbytes()
              for t in {t.untyped_storage().data_ptr(): t for t in tensors}.values())
out = {
    "model": "MiniCPM3-4B", "prefill_len": L,
    "num_cache_tensors": len(tensors),
    "shapes": sorted({str(tuple(t.shape)) for t in tensors}),
    "logical_bytes": logical, "logical_per_token": logical / L,
    "storage_bytes": storage, "storage_per_token": storage / L,
    "note": "storage>logical：V 是 kv_b_proj 输出（128 维/头）的非连续切片视图，钉住整块母张量",
}
res = Path(__file__).resolve().parent / "results"
res.mkdir(exist_ok=True)
(res / "e2b_cache_introspect.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
