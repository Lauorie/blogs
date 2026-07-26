"""抓现行：无缓存生成的第 51 步，把前 50 个 token 的 K/V 原样重算了一遍。

用法：python redundancy_proof.py /path/to/model
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()

text = "The history of computing is a story of layered abstractions. " * 8
ids = tok(text, return_tensors="pt").input_ids[:, :51].cuda()

with torch.inference_mode():
    # "第 50 步"：朴素前向 50 个 token。past_key_values 里正是这 50 个位置每一层的 K/V
    kv50 = model(input_ids=ids[:, :50], past_key_values=DynamicCache(),
                 use_cache=True).past_key_values
    # "第 51 步"：朴素做法把 51 个 token 又整段前向一遍——前 50 个位置全部重算
    kv51 = model(input_ids=ids, past_key_values=DynamicCache(),
                 use_cache=True).past_key_values

n_layers = len(kv50.layers)
same = sum(
    torch.equal(kv50.layers[l].keys, kv51.layers[l].keys[:, :, :50])
    and torch.equal(kv50.layers[l].values, kv51.layers[l].values[:, :, :50])
    for l in range(n_layers)
)
print(f"第 51 步重算出的前 50 个位置的 K/V，与第 50 步的结果逐位相同的层数：{same}/{n_layers}")
print(f"这 {n_layers} 层 × 50 个位置的重算，只为得到 1 个新位置的 K/V。")
