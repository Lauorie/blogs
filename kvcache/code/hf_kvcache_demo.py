"""亲手测一次 KV Cache 的加速：同一段 prompt，带 / 不带缓存各生成 1024 个 token。

用法：python hf_kvcache_demo.py /path/to/model
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel


@torch.inference_mode()
def generate(
    model: PreTrainedModel, prompt_ids: torch.Tensor, n: int, use_cache: bool
) -> tuple[list[int], float]:
    """贪心生成 n 个 token，返回 (token 列表, 耗时秒)。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps: list[torch.Tensor] = []  # token 先留在 GPU，计时结束后统一取回，避免逐步同步
    if use_cache:
        cache = DynamicCache()
        out = model(input_ids=prompt_ids, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        steps.append(nxt)
        for _ in range(n - 1):
            # 关键：每步只喂最新的 1 个 token，其余全部来自 cache
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            steps.append(nxt)
    else:
        seq = prompt_ids
        for _ in range(n):
            # 每步都把整段序列重算一遍。logits_to_keep=1 让 LM head 只算最后一个
            # 位置，保证两条路径口径一致（transformers v5；旧版叫 num_logits_to_keep）
            logits = model(input_ids=seq, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            steps.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return torch.cat(steps, dim=1)[0].tolist(), dt


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()

    prompt_ids = tok("The key idea of dynamic programming is", return_tensors="pt").input_ids.cuda()
    generate(model, prompt_ids, 8, use_cache=True)  # 预热 CUDA kernel

    n = 1024  # 上下文越长，无缓存路径越吃亏；改小到 256 加速几乎消失，见正文讨论
    fast, t_fast = generate(model, prompt_ids, n, use_cache=True)
    slow, t_slow = generate(model, prompt_ids, n, use_cache=False)

    print(f"生成 {n} token：有缓存 {t_fast:.2f}s（{n / t_fast:.1f} tok/s）"
          f"，无缓存 {t_slow:.2f}s（{n / t_slow:.1f} tok/s），加速 {t_slow / t_fast:.2f}x")
    same = sum(1 for a, b in zip(fast, slow) if a == b)
    prefix = next((i for i, (a, b) in enumerate(zip(fast, slow)) if a != b), n)
    print(f"两种方式输出一致的 token：{same}/{n}（前 {prefix} 个完全相同）")
    if prefix < n:
        # bf16 下两条路径走不同的 kernel，logits 有约 0.1~0.4 的浮点噪声；
        # 遇到概率接近的两个候选 token 时，argmax 可能翻转——数学上仍是等价的，
        # 换成 float32 重跑（去掉 dtype 参数）即可看到逐 token 一致。
        print("分叉处上下文：", repr(tok.decode(fast[max(0, prefix - 8): prefix + 2])))


if __name__ == "__main__":
    main()
