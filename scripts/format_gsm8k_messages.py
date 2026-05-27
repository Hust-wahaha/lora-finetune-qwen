"""
Build final chat-format GSM8K dataset from generated split records.

Input records are expected to live under data/interim/gsm8k/{train,test}.jsonl
and look like:
  {
    "id": "gsm8k_train_151",
    "source": "gsm8k",
    "modern_question": "...",
    "modern_cot": "...",
    "classical_question": "...",
    "classical_cot": "...",
    "structured_cot": "...",
    "answer": "283"
  }

Output records follow data/example.jsonl and are split by data_type:
  base_id, source, family, split, answer, dataset_variant, think_style,
  id, view, messages

Example:
  python scripts/format_gsm8k_messages.py ^
    --input data/interim/gsm8k ^
    --output-dir data/final/gsm8k_think ^
    --data-types modern classical modern2classical modern2structure
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFAULT_INPUT = os.path.join(DATA_DIR, "interim", "gsm8k")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR,"final","gsm8k")

SUPPORTED_DATA_TYPES = {
    "modern",
    "classic",
    "classical",
    "structure",
    "structured",
    "modern2classic",
    "modern2classical",
    "modern2structure",
    "modern2structured",
    "classic2modern",
    "classical2modern",
    "classic2structure",
    "classical2structure",
    "classic2structured",
    "classical2structured",
}

STYLE_NAME = {
    "modern": "白话文",
    "classical": "文言文",
    "structured": "结构化方式",
}

QUESTION_FIELD = {
    "modern": "modern_question",
    "classical": "classical_question",
}

THINK_FIELD = {
    "modern": "modern_think",
    "classical": "classical_think",
    "structured": "structured_think",
}


def read_jsonl(path: str) -> Iterable[tuple[int, dict | None, str | None]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no - 1, json.loads(line), None
            except json.JSONDecodeError as exc:
                yield line_no - 1, None, f"Invalid JSON on line {line_no}: {exc}"


def normalize_view(view: str) -> str:
    if view in ("classic", "classical"):
        return "classical"
    if view in ("structure", "structured"):
        return "structured"
    return view


def parse_data_type(data_type: str) -> tuple[str, str]:
    if data_type not in SUPPORTED_DATA_TYPES:
        choices = ", ".join(sorted(SUPPORTED_DATA_TYPES))
        raise ValueError(f"Unsupported data_type '{data_type}'. Choices: {choices}")

    if "2" in data_type:
        question_view, think_view = data_type.split("2", 1)
    else:
        question_view = data_type
        think_view = data_type

    question_view = normalize_view(question_view)
    think_view = normalize_view(think_view)

    if question_view == "structured":
        raise ValueError(
            f"Unsupported data_type '{data_type}': structured can only be used as think style."
        )

    return question_view, think_view


def checkpoint_index(item: dict, fallback_idx: int) -> str:
    raw_id = str(item.get("id", ""))
    match = re.search(r"(\d+)$", raw_id)
    if match:
        return str(int(match.group(1)))
    return str(fallback_idx)


def dataset_item_id(item: dict, fallback_idx: int, dataset: str) -> str:
    raw_id = item.get("id")
    if raw_id:
        return str(raw_id)
    return f"{dataset}-{fallback_idx}"


def get_question(item: dict, question_view: str) -> str:
    field = QUESTION_FIELD[question_view]
    question = item.get(field, "")
    if not question:
        raise ValueError(f"Missing required field '{field}' for item {item.get('id')}")
    return question


def get_think(item: dict, think_view: str) -> str:
    item_with_aliases = {
        **item,
        "modern_think": item.get("modern_think") or item.get("modern_cot", ""),
        "classical_think": item.get("classical_think") or item.get("classical_cot", ""),
        "structured_think": item.get("structured_think") or item.get("structured_cot", ""),
    }
    field = THINK_FIELD[think_view]
    think = item_with_aliases.get(field, "")
    if not think:
        raise ValueError(f"Missing required field '{field}' for item {item.get('id')}")
    return think


def build_system_message(think_view: str) -> str:
    style = STYLE_NAME[think_view]
    return (
        "你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。"
        f"请用{style}进行思考，并将最终答案输出在字符'答案：'后面"
    )


def build_assistant_message(think: str, answer: str) -> str:
    return f"<think>\n{think}\n</think>\n\n答案：{answer}。"


def build_record(
    item: dict,
    *,
    dataset: str,
    split: str,
    family: str,
    idx: str,
    data_type: str,
) -> dict:
    question_view, think_view = parse_data_type(data_type)
    question = get_question(item, question_view)
    think = get_think(item, think_view)
    answer = str(item.get("answer", "")).strip()
    base_id = f"{dataset}-{idx}"

    record = {
        "base_id": base_id,
        "source": dataset,
        "family": family,
        "split": split,
    }

    for out_field, in_field in (
        ("modern_question", "modern_question"),
        ("classical_question", "classical_question"),
        ("modern_think", "modern_cot"),
        ("classical_think", "classical_cot"),
        ("structured_think", "structured_cot"),
    ):
        value = item.get(out_field) or item.get(in_field)
        if value:
            record[out_field] = value

    record.update(
        {
            "answer": answer,
            "dataset_variant": "think",
            "think_style": "by_view",
            "id": f"{base_id}-{data_type}",
            "view": data_type,
            "messages": [
                {"role": "system", "content": build_system_message(think_view)},
                {"role": "user", "content": question},
                {"role": "assistant", "content": build_assistant_message(think, answer)},
            ],
        }
    )
    return record


def build_dataset(
    input_path: str,
    output_dir: str,
    *,
    data_types: list[str],
    dataset: str,
    split: str,
    family: str,
    limit: int | None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    dataset_paths = {
        data_type: os.path.join(output_dir, f"{data_type}.jsonl")
        for data_type in data_types
    }
    metadata_path = os.path.join(output_dir, f"metadata.json")
    error_path = os.path.join(output_dir, f"error_ids.jsonl")
    error_id_path = os.path.join(output_dir, f"error_ids.txt")

    written = 0
    seen_items = 0
    success_items = 0
    error_items = []
    records_by_type = {data_type: 0 for data_type in data_types}

    output_files = {
        data_type: open(path, "w", encoding="utf-8")
        for data_type, path in dataset_paths.items()
    }
    try:
        for fallback_idx, item, read_error in read_jsonl(input_path):
            if limit is not None and seen_items >= limit:
                break
            seen_items += 1

            if read_error is not None:
                error_items.append(
                    {
                        "id": f"{dataset}-line-{fallback_idx + 1}",
                        "line_index": fallback_idx,
                        "error": read_error,
                    }
                )
                continue

            item_id = dataset_item_id(item, fallback_idx, dataset)
            try:
                idx = checkpoint_index(item, fallback_idx)
                built_records = []
                for data_type in data_types:
                    record = build_record(
                        item,
                        dataset=dataset,
                        split=split,
                        family=family,
                        idx=idx,
                        data_type=data_type,
                    )
                    built_records.append((data_type, record))

                for data_type, record in built_records:
                    output_files[data_type].write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    records_by_type[data_type] += 1
                success_items += 1
            except Exception as exc:
                error_items.append(
                    {
                        "id": item_id,
                        "line_index": fallback_idx,
                        "error": str(exc),
                    }
                )
                continue
    finally:
        for f in output_files.values():
            f.close()

    with open(error_path, "w", encoding="utf-8") as f:
        for error in error_items:
            f.write(json.dumps(error, ensure_ascii=False) + "\n")

    with open(error_id_path, "w", encoding="utf-8") as f:
        for error in error_items:
            f.write(f"{error['id']}\n")

    metadata = {
        "dataset": dataset,
        "family": family,
        "split": split,
        "dataset_variant": "think",
        "think_style": "by_view",
        "input_path": os.path.normpath(input_path),
        "output_dir": os.path.normpath(output_dir),
        "dataset_files": {
            data_type: os.path.basename(path)
            for data_type, path in dataset_paths.items()
        },
        "metadata_file": os.path.basename(metadata_path),
        "error_ids_file": os.path.basename(error_id_path),
        "error_details_file": os.path.basename(error_path),
        "data_types": data_types,
        "supported_data_types": sorted(SUPPORTED_DATA_TYPES),
        "limit": limit,
        "source_items_seen": seen_items,
        "source_items_success": success_items,
        "source_items_failed": len(error_items),
        "records_written": written,
        "records_by_type": records_by_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema": {
            "top_level_fields": [
                "base_id",
                "source",
                "family",
                "split",
                "answer",
                "dataset_variant",
                "think_style",
                "id",
                "view",
                "messages",
            ],
            "message_roles": ["system", "user", "assistant"],
            "assistant_format": "<think>\\n...\\n</think>\\n\\n答案：...",
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "dataset_paths": dataset_paths,
        "metadata_path": metadata_path,
        "error_path": error_path,
        "error_id_path": error_id_path,
        "written": written,
        "seen_items": seen_items,
        "success_items": success_items,
        "failed_items": len(error_items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GSM8K checkpoint records to final chat-format JSONL."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input dataset path. It contains train and test dataset JSONL")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder. It will contain dataset JSONL, metadata, and error ids.",
    )
    parser.add_argument("--dataset", default="gsm8k", help="Dataset name used in base_id/source.")
    # parser.add_argument("--split", default="train", help="Split value written to each record.")
    parser.add_argument("--family", default="native", help="Family value written to each record.")
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["modern", "classical", "modern2classical", "modern2structure"],
        help="Views to generate, e.g. modern classical modern2classical modern2structure.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit source items, not output rows.")
    args = parser.parse_args()

    for split in ['train', 'test']:
        print(f"Processing split '{split}'...")
        split_data_path = os.path.join(args.input, f"{split}.jsonl")
        summary = build_dataset(
            split_data_path,
            os.path.join(args.output_dir, split),
            data_types=args.data_types,
            dataset=args.dataset,
            split=split,
            family=args.family,
            limit=args.limit,
        )
        print(f"Wrote {summary['written']} records:")
        for data_type, path in summary["dataset_paths"].items():
            print(f"  {data_type}: {path}")
        print(f"Wrote metadata to {summary['metadata_path']}")
        print(f"Wrote {summary['failed_items']} error ids to {summary['error_id_path']}")
        print(f"Wrote error details to {summary['error_path']}")


if __name__ == "__main__":
    main()
