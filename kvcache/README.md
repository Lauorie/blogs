# KV Cache 博客工作区

`kvcache.md` 是最终博客正文，其余目录是它的全部支撑材料——文中每个数字都能在这里找到出处。

```
kvcache-workshop/
├── kvcache.md              # 博客正文（10 张图、3 段可运行代码、实测数据）
├── code/
│   ├── toy_kvcache.py      # 文中代码：从零实现带 KV Cache 的 attention（CPU 可跑）
│   ├── redundancy_proof.py # 文中代码："抓现行"——证明第 51 步逐位重算了前 50 份 K/V（需 GPU）
│   ├── hf_kvcache_demo.py  # 文中代码：HF transformers 实测加速（需 GPU）
│   ├── plot_figures.py     # 三张数据图的绘图脚本（读 experiments/results）
│   ├── draw_fig3.py        # 示意图 fig3 的绘制脚本（行数需精确，故不用文生图）
│   ├── draw_fig1b.py       # 示意图 fig1b（自回归循环，中文 token 需精确）的绘制脚本
│   └── restore_code_blocks.py  # 一键把正文代码块恢复为 code/ 正本（防编辑器智能标点损坏）
├── experiments/
│   ├── common.py           # 共享工具：加载模型、计时、结果落盘
│   ├── e1_equivalence.py   # E1 等价性：cache on/off 逐 token 比对（bf16 + fp32）
│   ├── e2_latency.py       # E2 延迟：每步耗时曲线 + 端到端 1024 token
│   ├── e3_ttft.py          # E3 TTFT：prefill 耗时 vs prompt 长度
│   ├── e4_memory.py        # E4 显存：KV Cache 实测 vs 公式
│   ├── run_all.sh          # 服务器上顺序执行四个实验
│   └── results/            # 原始结果 JSON、run.log、pip freeze、GPU 信息
└── figures/                # 全部图片；AI 生成的示意图附 .prompt.md 生成词（fig3 除外，见 draw_fig3.py）
```

## 复现

实验在一台 RTX 5090（32 GB）上运行，模型 Qwen2.5-3B（bf16）：

```bash
scp -r experiments <gpu-server>:~/kvcache-exp
ssh <gpu-server> 'bash ~/kvcache-exp/run_all.sh'   # 约 15 分钟
scp -r <gpu-server>:~/kvcache-exp/results experiments/results
python3 code/plot_figures.py                        # 重新出三张数据图
```

依赖：`torch`、`transformers>=5`（实验用 5.14.1）、`matplotlib`。绘图脚本的中文字体需要单文件版 Noto Sans CJK（`~/.fonts/NotoSansCJKsc-Regular.otf`，matplotlib 不支持 .ttc 集合字体）。
