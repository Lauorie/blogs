# MLA 博客工作区

`正文.md` 是最终博客，其余目录是全部支撑材料——文中每个数字都能在这里找到出处。

```
mla-workshop/
├── 正文.md                  # 博客正文（6 张图、1 段可运行代码、实测数据）
├── code/
│   ├── toy_mla.py           # 文中代码正本：从零实现 MLA 四连验证（fp64/CPU 可跑）
│   ├── draw_data_figs.py    # 数据图：账本（E3）与微基准（E4）
│   ├── draw_absorb.py       # 示意图：矩阵吸收"换括号"
│   ├── draw_schematics.py   # 示意图：速记压缩 / 解耦 RoPE
│   └── restore_code_blocks.py  # 一键把正文代码块恢复为正本（防编辑器智能标点损坏）
├── experiments/
│   ├── e2_realmodel.py      # 真机对账：MiniCPM3-4B（MLA）vs Qwen2.5-3B（GQA）
│   ├── e3_ledger.py         # 从各模型 config.json 算每 token KV 缓存账本
│   ├── e4_microbench.py     # 解压式 vs 吸收式 vs MHA 的单层解码微基准
│   └── results/             # 原始结果 JSON
└── figures/                 # 全部图片（fig_shorthand/fig_rope_lane 为文生图，附 .prompt.md；mha-gqa-mla.png 为 DeepSeek-V2 论文 Figure 3，出处见图注；其余 matplotlib）
```

## 复现

- 本地（CPU 即可）：`python code/toy_mla.py`；绘图 `python code/draw_*.py`
- GPU 服务器（RTX 5090，PyTorch 2.12）：把 `experiments/` 拷到服务器逐个运行。
  注意 MiniCPM3-4B 的 trust_remote_code 与 transformers v5 不兼容，需单独建
  transformers==4.46 的 venv 运行 `e2_realmodel.py MiniCPM3-4B`。
- 依赖：torch、transformers、huggingface_hub（e3 需联网取 config）、matplotlib。
  中文图字体需单文件版 Noto Sans CJK（`~/.fonts/NotoSansCJKsc-Regular.otf`）。
