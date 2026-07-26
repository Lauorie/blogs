"""从零实现带 KV Cache 的单头因果自注意力，并验证与整段重算逐位一致。

只依赖 torch，CPU 上一秒跑完：python toy_kvcache.py
"""

import torch
from torch import Tensor

torch.manual_seed(42)
D = 64  # 模型维度


class Attention:
    """单头因果自注意力（推理视角，省略 Wo 和多头拆分，逻辑不受影响）。"""

    def __init__(self, d: int) -> None:
        self.d = d
        self.Wq = torch.randn(d, d) / d**0.5
        self.Wk = torch.randn(d, d) / d**0.5
        self.Wv = torch.randn(d, d) / d**0.5

    def forward_full(self, x: Tensor) -> Tensor:
        """整段重算：x 形状 [n, d]，返回全部 n 个位置的输出。"""
        Q, K, V = x @ self.Wq, x @ self.Wk, x @ self.Wv
        scores = Q @ K.T / self.d**0.5                  # [n, n]
        causal = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(causal, float("-inf"))  # 因果 mask：只看左边
        return scores.softmax(-1) @ V                   # [n, d]

    def forward_step(self, x_t: Tensor, cache: dict[str, list[Tensor]]) -> Tensor:
        """增量计算：x_t 形状 [d]，只为最新 token 算 q/k/v，其余读缓存。"""
        q, k, v = x_t @ self.Wq, x_t @ self.Wk, x_t @ self.Wv
        cache["K"].append(k)                            # 本步唯一的新 K
        cache["V"].append(v)                            # 本步唯一的新 V
        K, V = torch.stack(cache["K"]), torch.stack(cache["V"])
        weights = (q @ K.T / self.d**0.5).softmax(-1)   # [t]，无需 mask：缓存里没有未来
        return weights @ V                              # [d]


def main() -> None:
    attn = Attention(D)
    xs = torch.randn(10, D)  # 假装是 10 个 token 的输入向量序列

    # 路径 A：模拟"无缓存"生成——每一步把前缀整段重算，只取最后一个位置
    full_outs = [attn.forward_full(xs[: t + 1])[-1] for t in range(len(xs))]

    # 路径 B：带 KV Cache——每一步只算最新 token
    cache: dict[str, list[Tensor]] = {"K": [], "V": []}
    step_outs = [attn.forward_step(xs[t], cache) for t in range(len(xs))]

    diff = max(float((a - b).abs().max()) for a, b in zip(full_outs, step_outs))
    print(f"两条路径输出的最大差值: {diff:.2e}")  # 浮点噪声级别
    assert diff < 1e-5, "KV Cache 与整段重算不一致！"
    print("验证通过：KV Cache 只是省去重复计算，数学上完全等价。")


if __name__ == "__main__":
    main()
