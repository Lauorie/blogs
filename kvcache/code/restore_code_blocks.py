"""把 kvcache.md 里的三个 Python 代码块恢复成 code/ 下正本的逐字节副本。

编辑器的"智能标点"会把代码里的英文直引号改成中文弯引号，导致代码无法运行。
任何时候发现代码块坏了，跑一次本脚本即可：python restore_code_blocks.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE.parent / "kvcache.md"
CANONICAL = ["redundancy_proof.py", "toy_kvcache.py", "hf_kvcache_demo.py"]


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"```python\n(.*?)```", md, re.DOTALL))
    assert len(blocks) == len(CANONICAL), f"预期 {len(CANONICAL)} 个代码块，实际 {len(blocks)}"
    for match, name in reversed(list(zip(blocks, CANONICAL))):
        md = md[: match.start(1)] + (HERE / name).read_text(encoding="utf-8") + md[match.end(1):]
    MD.write_text(md, encoding="utf-8")
    for block, name in zip(re.findall(r"```python\n(.*?)```", MD.read_text(encoding="utf-8"), re.DOTALL), CANONICAL):
        status = "OK" if block == (HERE / name).read_text(encoding="utf-8") else "FAILED"
        print(f"{name}: {status}")


if __name__ == "__main__":
    main()
