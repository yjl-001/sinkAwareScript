#!/usr/bin/env python3
"""不加载 torch/model 的 Weaver 插点策略单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "memgen/model/augmentation_strategy.py"


def load_strategy_module():
    """绕过 memgen.model 的重量级 __init__，只加载纯 Python 策略模块。"""

    spec = importlib.util.spec_from_file_location("memgen_augmentation_strategy_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load strategy module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    strategy_module = load_strategy_module()
    point = strategy_module.InferenceInsertionPoint
    build = strategy_module.build_weaver_insertion_strategy
    query_index = strategy_module.query_index_for_insertion_point

    assert query_index(4) == 3

    points = [
        point(index=4, is_delimiter=True, sink_score=0.31),
        point(index=5, is_delimiter=False, sink_score=0.90),
        point(index=6, is_delimiter=True, sink_score=0.20),
        point(index=7, is_delimiter=True, sink_score=0.80),
        point(index=8, is_delimiter=False, sink_score=0.40),
    ]

    first_k = build({"name": "first_k"})
    assert first_k.select(points, max_num=2) == [4, 6]

    candidate_sink = build(
        {"name": "candidate_sink_threshold", "sink_score_threshold": 0.30}
    )
    # 非 delimiter 的 0.90 不可进入候选池；结果恢复为因果位置顺序。
    assert candidate_sink.select(points, max_num=2) == [4, 7]

    sequence_sink = build(
        {"name": "sequence_sink_threshold", "sink_score_threshold": 0.30}
    )
    assert sequence_sink.select(points, max_num=2) == [5, 7]
    assert sequence_sink.select(points, max_num=0) == []

    strict_threshold = build(
        {"name": "candidate_sink_threshold", "sink_score_threshold": 0.80}
    )
    # 用户定义是 score > threshold，等于阈值的 0.80 必须被排除。
    assert strict_threshold.select(points, max_num=5) == []

    try:
        build({"name": "unknown"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown strategy should be rejected")

    try:
        build({"name": "first_k"}).select(points, max_num=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative budget should be rejected")

    print("[ok] Weaver insertion strategy tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
