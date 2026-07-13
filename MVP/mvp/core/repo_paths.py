"""独立 sinkAwareScript 仓库的路径工具。

MVP 可以从仓库根目录、上层 MemGen 目录或绝对路径启动。所有项目内相对路径
最终都归一到当前独立仓库，避免意外导入上层同名的 ``memgen``/``data`` 包。
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MVP_ROOT = PROJECT_ROOT / "MVP"

# 必须放在 sys.path 首位；仅判断“是否存在”可能让上层 MemGen 抢先被导入。
project_root_str = str(PROJECT_ROOT)
if not sys.path or sys.path[0] != project_root_str:
    sys.path = [item for item in sys.path if item != project_root_str]
    sys.path.insert(0, project_root_str)


def resolve_project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """把 CLI/YAML 中的路径解析为绝对路径。

    兼容迁移前的 ``sinkAwareScript/MVP/...`` 写法，但新配置统一使用
    ``MVP/...``。不存在的输出目录也会稳定落到 PROJECT_ROOT 下。
    """

    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        candidates = [Path.cwd() / path, PROJECT_ROOT / path]
        if path.parts and path.parts[0] == PROJECT_ROOT.name:
            candidates.append(PROJECT_ROOT.joinpath(*path.parts[1:]))
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        resolved = (existing or (PROJECT_ROOT / path)).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Project path does not exist: {resolved}")
    return resolved
