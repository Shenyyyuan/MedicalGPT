import ast
from pathlib import Path
from typing import Optional

from datasets import Dataset, concatenate_datasets


def _load_sft_helpers():
    source = Path(__file__).resolve().parents[1].joinpath(
        "training", "supervised_finetuning.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "iter_sft_sources",
        "sample_dataset",
        "build_mixed_train_dataset",
    }
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "Optional": Optional,
        "concatenate_datasets": concatenate_datasets,
    }
    exec(compile(module, "<sft_helpers>", "exec"), namespace)
    return namespace


def test_iter_sft_sources_accepts_huatuo_questions_answers():
    helpers = _load_sft_helpers()

    result = list(
        helpers["iter_sft_sources"](
            {
                "questions": ["心悸怎么办？"],
                "answers": ["建议完善心电图等检查。"],
            }
        )
    )

    assert result == [
        [
            {"from": "human", "value": "心悸怎么办？"},
            {"from": "gpt", "value": "建议完善心电图等检查。"},
        ]
    ]


def test_iter_sft_sources_ignores_none_aligned_conversations_column():
    helpers = _load_sft_helpers()

    result = list(
        helpers["iter_sft_sources"](
            {
                "instruction": ["高血压能喝牛奶吗？"],
                "output": ["可以适量饮用低脂牛奶。"],
                "conversations": [None],
                "questions": [None],
                "answers": [None],
            }
        )
    )

    assert result == [
        [
            {"from": "human", "value": "高血压能喝牛奶吗？"},
            {"from": "gpt", "value": "可以适量饮用低脂牛奶。"},
        ]
    ]


def test_build_mixed_train_dataset_uses_requested_ratio():
    helpers = _load_sft_helpers()
    domain = Dataset.from_list(
        [{"instruction": f"domain-{i}", "output": "a"} for i in range(10)]
    )
    general = Dataset.from_list(
        [{"questions": f"general-{i}", "answers": "b"} for i in range(10)]
    )

    mixed = helpers["build_mixed_train_dataset"](
        domain,
        general,
        total_samples=10,
        domain_ratio=0.8,
        seed=42,
    )

    assert len(mixed) == 10
    assert sum(str(row.get("instruction", "")).startswith("domain-") for row in mixed) == 8
    assert sum(str(row.get("questions", "")).startswith("general-") for row in mixed) == 2
