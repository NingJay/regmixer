#!/usr/bin/env python3
"""
精确统计 Nemotron-CC 各子源的 token 数量

使用方法：
    python scripts/count_tokens.py

输出：
    - 按充足度排序的 token 统计表格
    - 标识数据不足的子源（< 3B tokens）
"""

import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq


# 数据路径基准
BASE_PATH = "/home/staff/jiayining/Datasets/OpenSeek-Pretrain-100B/Datasets/OpenSeek-Pretrain-100B_text"

# 11 个子源的路径映射（与 nemotron-cc-round1-3b.yaml 一致）
SOURCE_PATTERNS = {
    "actual:high": f"{BASE_PATH}/Nemotron-CC-high-actual-actual-high_part_*_text_document/train-*.parquet",
    "actual:medium_high": f"{BASE_PATH}/Nemotron-CC-medium-actual-actual-high_part_*_text_document/train-*.parquet",
    "actual:medium": f"{BASE_PATH}/Nemotron-CC-high-actual-actual-mid_part_*_text_document/train-*.parquet",
    "actual:medium_low": f"{BASE_PATH}/Nemotron-CC-medium-actual-actual-low_part_*_text_document/train-*.parquet",
    "actual:low": f"{BASE_PATH}/Nemotron-CC-high-actual-actual-low_part_*_text_document/train-*.parquet",
    "distill": f"{BASE_PATH}/Nemotron-CC-high-synthetic-distill-high_part_*_text_document/train-*.parquet",
    "diverse_qa_pairs": f"{BASE_PATH}/Nemotron-CC-high-synthetic-diverse_qa_pairs-high_part_*_text_document/train-*.parquet",
    "extract_knowledge": f"{BASE_PATH}/Nemotron-CC-high-synthetic-extract_knowledge-high_part_*_text_document/train-*.parquet",
    "knowledge_list": f"{BASE_PATH}/Nemotron-CC-high-synthetic-knowledge_list-high_part_*_text_document/train-*.parquet",
    "wrap_medium:high": f"{BASE_PATH}/Nemotron-CC-high-synthetic-wrap_medium-high_part_*_text_document/train-*.parquet",
    "wrap_medium:low": f"{BASE_PATH}/Nemotron-CC-low-synthetic-wrap_medium-low_part_*_text_document/train-*.parquet",
}

# 训练需求
REQUIRED_TOKENS = 3_000_000_000  # 3B tokens


def count_tokens_from_parquet(file_path: str) -> int:
    """
    从单个 Parquet 文件统计 token 数量（读取 length 列）

    Args:
        file_path: Parquet 文件路径

    Returns:
        该文件的总 token 数
    """
    try:
        pf = pq.ParquetFile(file_path)
        total_tokens = 0

        # 遍历所有 row groups
        for row_group_idx in range(pf.num_row_groups):
            table = pf.read_row_group(row_group_idx, columns=['length'])
            total_tokens += table['length'].to_pandas().sum()

        return int(total_tokens)
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {file_path}", file=sys.stderr)
        print(f"        {e}", file=sys.stderr)
        return 0


def count_tokens_for_source(pattern: str) -> Tuple[int, int]:
    """
    统计某个子源的 token 数量

    Args:
        pattern: Glob 路径模式

    Returns:
        (文件数, 总 token 数)
    """
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARNING] 未找到匹配文件: {pattern}", file=sys.stderr)
        return 0, 0

    total_tokens = 0
    for file_path in files:
        tokens = count_tokens_from_parquet(file_path)
        total_tokens += tokens

    return len(files), total_tokens


def format_sufficiency(tokens: int, required: int) -> str:
    """
    格式化充足度标识

    Args:
        tokens: 实际 token 数
        required: 需求 token 数

    Returns:
        格式化的充足度字符串（带 emoji）
    """
    ratio = tokens / required if required > 0 else 0
    percentage = int(ratio * 100)

    if ratio >= 1.0:
        return f"✅ {percentage}%"
    elif ratio >= 0.9:
        return f"⚠️  {percentage}%"
    else:
        return f"❌ {percentage}%"


def main():
    print("=" * 80)
    print("Nemotron-CC 子源 Token 数量统计（精确计数）")
    print("=" * 80)
    print(f"方法: 读取 Parquet length 列（100% 准确）")
    print(f"需求: {REQUIRED_TOKENS / 1e9:.1f}B tokens per source")
    print("=" * 80)
    print()

    # 统计所有子源
    results: List[Tuple[str, int, int]] = []

    for source_name, pattern in SOURCE_PATTERNS.items():
        print(f"[统计] {source_name:<20} ...", end=" ", flush=True)
        num_files, total_tokens = count_tokens_for_source(pattern)
        results.append((source_name, num_files, total_tokens))
        print(f"{num_files:2d} files, {total_tokens / 1e9:.2f}B tokens")

    # 按 token 数排序（从小到大）
    results.sort(key=lambda x: x[2])

    # 输出格式化表格
    print()
    print("=" * 80)
    print("统计结果（按 token 数排序）")
    print("=" * 80)
    print(f"{'Source':<20} | {'Files':<6} | {'Tokens(B)':<10} | {'充足度 (vs 3B)':<15}")
    print("-" * 80)

    total_files = 0
    total_tokens = 0
    insufficient_sources = []

    for source_name, num_files, tokens in results:
        tokens_in_billions = tokens / 1e9
        sufficiency = format_sufficiency(tokens, REQUIRED_TOKENS)

        print(f"{source_name:<20} | {num_files:<6d} | {tokens_in_billions:<10.2f} | {sufficiency:<15}")

        total_files += num_files
        total_tokens += tokens

        if tokens < REQUIRED_TOKENS:
            insufficient_sources.append(source_name)

    print("-" * 80)
    print(f"{'Total':<20} | {total_files:<6d} | {total_tokens / 1e9:<10.2f} | {total_tokens / (REQUIRED_TOKENS * 11):.2f}x (总和)")
    print("=" * 80)

    # 分析结论
    print()
    print("⚠️  数据充足度分析")
    print("=" * 80)

    min_source = results[0]
    print(f"最小数据集: {min_source[0]}")
    print(f"Token 数量: {min_source[2] / 1e9:.2f}B tokens")
    print(f"充足度: {min_source[2] / REQUIRED_TOKENS * 100:.1f}%")
    print()

    if insufficient_sources:
        print(f"数据不足的子源（< 3B）: {len(insufficient_sources)}/{len(SOURCE_PATTERNS)}")
        for source in insufficient_sources:
            print(f"  - {source}")
        print()
        print("✅ 建议措施:")
        print("   1. 配置中设置 allow_repetition: true（允许数据循环）")
        print("   2. 或调整 minimum_weight 避免小数据集被分配高权重")
    else:
        print("✅ 所有子源数据充足，可以设置 allow_repetition: false")

    print("=" * 80)


if __name__ == "__main__":
    main()
