"""把博客正文里的 ```python 代码块恢复成 code/ 正本的逐字节副本，并校验一致。

背景：某些编辑器的"智能标点"会把代码块里的英文直引号改成中文弯引号，
Python 直接语法错误。把本脚本拷进项目 code/ 目录，改好下面两个常量即可。

用法：python restore_code_blocks.py [--check]
    --check 只校验不修改（用于交付前检查清单）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# ↓ 按项目修改：正文路径，以及正文中代码块的出现顺序对应的正本文件名
MD = HERE.parent / "正文.md"
CANONICAL = ["toy_mla.py"]


def main() -> int:
    check_only = "--check" in sys.argv
    md = MD.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"```python\n(.*?)```", md, re.DOTALL))
    if len(blocks) != len(CANONICAL):
        print(f"预期 {len(CANONICAL)} 个代码块，实际 {len(blocks)}——先更新 CANONICAL 列表")
        return 1

    dirty = False
    for match, name in zip(blocks, CANONICAL):
        if match.group(1) != (HERE / name).read_text(encoding="utf-8"):
            dirty = True
            print(f"{name}: DIFFERS")
        else:
            print(f"{name}: OK")

    if not dirty or check_only:
        return 1 if (dirty and check_only) else 0

    for match, name in reversed(list(zip(blocks, CANONICAL))):
        md = md[: match.start(1)] + (HERE / name).read_text(encoding="utf-8") + md[match.end(1):]
    MD.write_text(md, encoding="utf-8")
    print("已恢复；重新运行 --check 验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
