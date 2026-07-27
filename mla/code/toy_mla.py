"""从零实现 MLA（Multi-head Latent Attention）的推理视角，验证四件事：

A vs B：只缓存 latent、用时解压  ==  缓存完整 K/V（逐位一致）
A vs C：矩阵吸收（不解压、直接在 latent 空间做 attention）==  显式解压
D    ：把 RoPE 夹在中间再天真吸收 → 数值直接不等（这就是为什么需要解耦 RoPE）
E    ：解耦 RoPE（k_R 单独缓存一小截）→ 恢复一致

维度取 DeepSeek-V2 真实值（hidden 5120、128 头 × head_dim 128、d_c=512、d_R=64），
权重随机、fp64、CPU 可跑：python toy_mla.py
"""

import torch

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

D, H, DH, DC, DR = 5120, 128, 128, 512, 64   # hidden / 头数 / 每头维 / latent 维 / RoPE 维
T = 32                                        # 序列长度

# 权重：降维 W_dkv，升维 W_uk/W_uv（按头拆），查询 W_q，解耦 RoPE 的 W_kr/W_qr
W_dkv = torch.randn(D, DC) / D**0.5
W_uk = torch.randn(H, DC, DH) / DC**0.5
W_uv = torch.randn(H, DC, DH) / DC**0.5
W_q = torch.randn(H, D, DH) / D**0.5
W_kr = torch.randn(D, DR) / D**0.5
W_qr = torch.randn(H, D, DR) / D**0.5

h = torch.randn(T, D)                         # 假装是 T 个 token 的 hidden state
c = h @ W_dkv                                 # latent：每 token 只有 512 维 [T, DC]
q = torch.einsum("td,hde->the", h, W_q)       # [T, H, DH]


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """标准 RoPE：最后一维按两两一组旋转，角度随位置 pos 变化。"""
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half) / half))
    ang = pos[:, None] * freq[None, :]                      # [T, half]
    cos, sin = ang.cos(), ang.sin()
    while cos.dim() < x.dim():                              # 对齐 [T, ..., half]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attn(scores: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """给定打分 [H, T]（最新 token 对全部位置）和 V [T, H, DH]，返回各头输出。"""
    w = (scores / (DH + DR) ** 0.5).softmax(-1)
    return torch.einsum("ht,the->he", w, v)


pos = torch.arange(T)
diff = lambda a, b: float((a - b).abs().max())

# ---- 路径 A：显式缓存完整 K/V（不含 RoPE，先看纯低秩部分）----
K = torch.einsum("tc,hce->the", c, W_uk)      # 解压出每头的 K [T, H, DH]
V = torch.einsum("tc,hce->the", c, W_uv)
score_A = torch.einsum("he,the->ht", q[-1], K)
out_A = attn(score_A, V)                      # 最新 token 的各头输出 [H, DH]

# ---- 路径 B：只缓存 c，用时解压（与 A 是同一段数学）----
K_b = torch.einsum("tc,hce->the", c, W_uk)
score_B = torch.einsum("he,the->ht", q[-1], K_b)
out_B = attn(score_B, torch.einsum("tc,hce->the", c, W_uv))
print(f"A(缓存完整K/V) vs B(缓存latent+解压): 最大差值 {diff(out_A, out_B):.1e}")

# ---- 路径 C：矩阵吸收——把 W_uk 吸进 q，attention 直接在 512 维 latent 上做 ----
q_tilde = torch.einsum("he,hce->hc", q[-1], W_uk)   # 吸收后的查询 [H, DC]
score_C = torch.einsum("hc,tc->ht", q_tilde, c)     # 打分只用 c，不解压 K
w_C = (score_C / (DH + DR) ** 0.5).softmax(-1)
u = torch.einsum("ht,tc->hc", w_C, c)               # 加权 latent [H, DC]
out_C = torch.einsum("hc,hce->he", u, W_uv)         # 最后才升维（可再吸进 W_O）
print(f"A(显式解压)     vs C(矩阵吸收):      最大差值 {diff(out_A, out_C):.1e}")

# ---- D：天真吸收 + RoPE：把 RoPE 加在解压后的 K 上，再假装还能吸收 ----
K_rope = rope(K, pos)                                # 正确做法：旋转作用在每头 K 上
score_true = torch.einsum("he,the->ht", rope(q, pos)[-1], K_rope)
c_rope = rope(c, pos)                                # 天真做法：把旋转挪到 latent 上
score_naive = torch.einsum("hc,tc->ht", torch.einsum("he,hce->hc", rope(q, pos)[-1], W_uk), c_rope)
print(f"D 天真吸收+RoPE vs 真值:            最大差值 {diff(score_true, score_naive):.1e}  <- 翻车")

# ---- E：解耦 RoPE——位置信息走单独的小通道 k_R（64 维，所有头共享），与 c 并排缓存 ----
k_r = rope(h @ W_kr, pos)                            # [T, DR]，这一截才带位置
q_r = rope(torch.einsum("td,hde->the", h, W_qr), pos)[-1]   # [H, DR]
score_E = score_C + torch.einsum("he,te->ht", q_r, k_r)     # 内容分 + 位置分
out_E = torch.einsum("hc,hce->he", torch.einsum("ht,tc->hc",
        (score_E / (DH + DR) ** 0.5).softmax(-1), c), W_uv)
# 对照：同一解耦结构、但显式解压 K/V 的算法
score_ref = torch.einsum("he,the->ht", q[-1], K) + torch.einsum("he,te->ht", q_r, k_r)
out_ref = attn(score_ref, V)
print(f"E 解耦RoPE吸收  vs 显式解压对照:     最大差值 {diff(out_E, out_ref):.1e}")

# ---- 缓存账本（每 token 每层，bf16 2 字节）----
mha, mla = 2 * H * DH * 2, (DC + DR) * 2
print(f"每 token 每层缓存: MHA {mha} B | MLA {mla} B -> {mha / mla:.1f}x 压缩")
