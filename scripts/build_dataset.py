"""
数学应用题数据集构建脚本
==========================
功能:
  1. GSM8K: 英语题目 + CoT + 答案 → DeepSeek API → 白话文题目, 白话文CoT, 文言文题目, 文言文CoT
  2. Math23k: 中文题目 + 公式(不给答案) → DeepSeek API → 白话文CoT, 文言文题目, 文言文CoT
     - API 输出答案置于 \boxed{} 中
     - 用正则提取答案并与原始答案比对，不一致则警告并记录 ID

使用方式:
  1. 设置环境变量 DEEPSEEK_API_KEY=你的API密钥
  2. python scripts/build_dataset.py [--dry-run] [--limit N] [--ratio 8:2] [--source gsm8k|math23k|all] [--output-dir data/interim]

输出:
  data/llm_output/{dataset}/checkpoint_{split}.jsonl  — 处理断点
  data/llm_output/{dataset}/{split}.jsonl             — split 数据集
  data/llm_output/{dataset}/mismatch_log.txt          — 答案不一致记录
"""

import json
import math
import os
import re
import sys
import time
import argparse
from json.decoder import JSONDecoder
from typing import Optional

import pandas as pd

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ============================================================================
# 配置
# ============================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

# 项目根目录 (脚本在 scripts/ 下)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/interim")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data/raw")

# 默认输出根目录。每个数据集会写入该目录下的同名子文件夹。
DEFAULT_OUTPUT_DIR = DATA_DIR

# API 调用间隔（秒），避免触发频率限制
API_CALL_INTERVAL = 0.5


# ============================================================================
# 数据加载
# ============================================================================

def load_gsm8k(split: str = "train") -> pd.DataFrame:
    """加载 GSM8K 主数据集（parquet 格式）。"""
    path = os.path.join(RAW_DATA_DIR, "gsm8k", "main", f"{split}-00000-of-00001.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"GSM8K 数据文件不存在: {path}")
    df = pd.read_parquet(path)
    df["id"] = [f"gsm8k_{split}_{i}" for i in range(len(df))]
    df["source"] = "gsm8k"
    return df


def load_math23k(split: str = "train") -> list:
    """加载 Math23k JSON 文件（pretty-printed 多对象格式）。"""
    filename = f"math23k_{split}.json"
    path = os.path.join(RAW_DATA_DIR, "Math23k", filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Math23k 数据文件不存在: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    decoder = JSONDecoder()
    data = []
    pos = 0
    raw_str = raw.decode("utf-8")
    while pos < len(raw_str):
        while pos < len(raw_str) and raw_str[pos] in " \t\r\n":
            pos += 1
        if pos >= len(raw_str):
            break
        try:
            obj, end = decoder.raw_decode(raw_str, pos)
            obj["source"] = "math23k"
            obj["split"] = split
            # 为 ID 添加前缀避免与其他数据源冲突
            obj["uid"] = f"math23k_{split}_{obj['id']}"
            data.append(obj)
            pos = end
        except json.JSONDecodeError:
            break
    return data


# ============================================================================
# 答案处理
# ============================================================================

def extract_gsm8k_final_answer(answer_text: str) -> Optional[str]:
    """从 GSM8K 的 answer 字段中提取最终答案（#### 之后的内容）。"""
    if not answer_text:
        return None
    parts = answer_text.split("####")
    if len(parts) > 1:
        return parts[-1].strip()
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """从文本中提取 \\boxed{...} 中的答案，支持嵌套大括号。"""
    # 匹配 \boxed{...}，处理一层嵌套
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # 取最后一个匹配（最终答案）
    return None


def normalize_answer(ans) -> str:
    """标准化答案以便比较：
    - 去空格、逗号
    - 整数去掉尾部 .0
    - 统一为字符串
    """
    s = str(ans).strip().replace(",", "").replace(" ", "").replace("%", "")
    try:
        f = float(s)
        if f == int(f):
            s = str(int(f))
    except ValueError:
        pass
    return s


def parse_ratio(ratio: str) -> tuple[int, int]:
    """解析 train:test 比例，例如 8:2。"""
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", ratio)
    if not match:
        raise ValueError("ratio 必须使用 train:test 格式，例如 8:2")
    train_part = int(match.group(1))
    test_part = int(match.group(2))
    if train_part <= 0 or test_part < 0:
        raise ValueError("ratio 中 train 必须大于 0，test 不能小于 0")
    return train_part, test_part


def split_limit_for(split: str, train_limit: Optional[int], ratio: str) -> Optional[int]:
    """根据 train limit 和 train:test ratio 计算每个 split 的处理数量。"""
    if train_limit is None:
        return None
    train_part, test_part = parse_ratio(ratio)
    if split == "train":
        return train_limit
    if split == "test":
        if test_part == 0:
            return 0
        return max(1, math.ceil(train_limit * test_part / train_part))
    return train_limit


def format_limit(limit: Optional[int]) -> str:
    if limit is None:
        return "全部"
    return str(limit)


def dataset_output_paths(output_dir: str, dataset: str, split: str) -> dict:
    """返回某个数据集 split 的输出路径集合。"""
    dataset_dir = os.path.join(output_dir, dataset)
    return {
        "dataset_dir": dataset_dir,
        "split_file": os.path.join(dataset_dir, f"{split}.jsonl"),
        "checkpoint": os.path.join(dataset_dir, f"checkpoint_{split}.jsonl"),
        "failed": os.path.join(dataset_dir, f"checkpoint_{split}.jsonl.failed"),
        "mismatch_log": os.path.join(dataset_dir, "mismatch_log.txt"),
    }


def write_dataset_metadata(dataset_dir: str, dataset: str, args, split_summaries: list) -> None:
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    metadata = {
        "dataset": dataset,
        "output_dir": dataset_dir,
        "source": args.source,
        "splits": args.splits,
        "train_limit": args.limit,
        "ratio": args.ratio,
        "resume": not args.no_resume,
        "model": MODEL_NAME,
        "split_summaries": split_summaries,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  写入 metadata: {metadata_path}")


# ============================================================================
# Prompt 模板
# ============================================================================

GSM8K_SYSTEM_PROMPT = """你是一个专业的数学题目翻译与文言文转换助手。你需要将英文数学应用题及其解答过程翻译转换为中文白话文和文言文两种风格。请严格遵循输出格式。"""


def build_gsm8k_user_prompt(question: str, answer: str, final_answer: str) -> str:
    """构建 GSM8K 的 API 请求 prompt。"""
    return f"""请处理以下英文数学应用题。给出题目、逐步解答过程和最终答案后，请生成对应的中文版本。

【英文题目】
{question}

【英文解答】
{answer}

【最终答案】
{final_answer}

请按以下格式严格输出四个部分：

【白话文题目】
（将题目翻译为现代白话文）

【白话文CoT】
（将解答过程翻译为现代白话文思维链）

【文言文题目】
（将题目改写为文言文风格）

【文言文CoT】
（将解答过程改写为文言文思维链）

【结构化CoT】
（将解答过程压缩成精简结构化表达的思维链）

注意：
1.文言文应使用文言句式、词汇，例如"之乎者也"等文言虚词；
2.数学表达式保留原样，且不要使用latex格式，直接用文本表达，例如 x + 2 = 5。"""


MATH23K_SYSTEM_PROMPT = """你是一个专业的数学解题与文言文转换助手。你需要用中文白话文解答数学应用题，并将题目和解答转换为文言文风格。答案必须放在 \\boxed{} 中，并且其中只含有数字，不包含单位。请严格遵循输出格式。"""


def build_math23k_user_prompt(original_text: str, equation: str) -> str:
    """构建 Math23k 的 API 请求 prompt（不给答案，让 API 自己算）。"""
    return f"""请处理以下中文数学应用题。已知题目和公式，请解答并转换为文言文。

【题目】
{original_text}

【公式】
{equation}

请按以下格式严格输出三个部分：

【白话文CoT】
（用现代白话文写出逐步解答的思维链）

【文言文题目】
（将题目改写为文言文风格）

【文言文CoT】
（用文言文风格写出逐步解答的思维链）

注意：文言文应使用文言句式、词汇，例如"之乎者也"等文言虚词，但数学表达式保留原样。"""


# ============================================================================
# API 调用
# ============================================================================

def create_client() -> OpenAI:
    """创建 DeepSeek API 客户端。"""
    if OpenAI is None:
        raise ModuleNotFoundError("未安装 openai 包，请先安装后再调用 API。")
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "请设置环境变量 DEEPSEEK_API_KEY。\n"
            "获取 API Key: https://platform.deepseek.com/api_keys"
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_api(client: OpenAI, system_prompt: str, user_prompt: str,
             max_retries: int = 5) -> str:
    """调用 DeepSeek API，带指数退避重试。"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)  # 最长等待 60 秒
                print(f"  ⚠ API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                print(f"     {wait}秒后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"API调用失败，已达最大重试次数 ({max_retries}): {last_error}"
                ) from last_error

    raise RuntimeError(f"API调用失败: {last_error}")


# ============================================================================
# 响应解析
# ============================================================================

SECTION_MARKERS = {
    "vernacular_question": "【白话文题目】",
    "vernacular_cot": "【白话文CoT】",
    "classical_question": "【文言文题目】",
    "classical_cot": "【文言文CoT】",
}

# 标记到 Section 的映射顺序
MARKER_ORDER = [
    ("vernacular_question", "【白话文题目】"),
    ("vernacular_cot", "【白话文CoT】"),
    ("classical_question", "【文言文题目】"),
    ("classical_cot", "【文言文CoT】"),
    ("structured_cot", "【结构化CoT】"),
]

MATH23K_MARKER_ORDER = [
    ("vernacular_cot", "【白话文CoT】"),
    ("classical_question", "【文言文题目】"),
    ("classical_cot", "【文言文CoT】"),
]


def parse_response(text: str, marker_list: list) -> dict:
    """
    根据标记列表解析 API 返回的文本，提取各部分内容。
    支持标记后紧跟冒号、换行等常见格式变体。

    Args:
        text: API 返回的原始文本
        marker_list: [(key, marker_string), ...] 的列表，按出现顺序

    Returns:
        dict: {key: content, ...}
    """
    result = {}
    for i, (key, marker) in enumerate(marker_list):
        # 找到当前标记的位置（支持全角/半角变体）
        start_idx = -1
        for variant in (marker, marker.replace("【", "[").replace("】", "]"),
                        marker.replace("：", ":")):
            pos = text.find(variant)
            if pos != -1:
                start_idx = pos
                marker_len = len(variant)
                break
        if start_idx == -1:
            result[key] = ""
            continue

        content_start = start_idx + marker_len
        # 跳过紧跟的冒号和空白
        while content_start < len(text) and text[content_start] in "：: \t\r\n":
            content_start += 1

        # 找到下一个标记的位置作为结束
        end_idx = len(text)
        for next_key, next_marker in marker_list[i + 1:]:
            for variant in (next_marker, next_marker.replace("【", "[").replace("】", "]")):
                pos = text.find(variant, content_start)
                if pos != -1:
                    end_idx = min(end_idx, pos)

        content = text[content_start:end_idx].strip()
        result[key] = content

    return result


# ============================================================================
# GSM8K 处理
# ============================================================================

def process_gsm8k(client: OpenAI, df: pd.DataFrame, checkpoint_path: str,
                  failed_path: str, split: str, limit: Optional[int] = None,
                  resume: bool = True) -> tuple:
    """处理 GSM8K 数据集，返回 (结果列表, 成功数, 失败数)。"""
    results = []
    processed_ids = set()

    # 断点续传：读取已处理的数据
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    processed_ids.add(obj["id"])
                    results.append(obj)
        print(f"[GSM8K] 从断点恢复，已处理 {len(processed_ids)} 条")

    total = len(df)
    success = len(processed_ids)
    fail = 0

    for idx, row in df.iterrows():
        item_id = row["id"]

        if item_id in processed_ids:
            continue

        if limit is not None and success >= limit:
            break

        question = row["question"]
        answer_text = row["answer"]
        final_answer = extract_gsm8k_final_answer(answer_text)

        user_prompt = build_gsm8k_user_prompt(question, answer_text, final_answer or "N/A")

        print(f"\n[GSM8K] 处理第 {success+1}/{min(total, limit or total)} 条 (id={item_id})...")

        try:
            response = call_api(client, GSM8K_SYSTEM_PROMPT, user_prompt)
            parsed = parse_response(response, MARKER_ORDER)
            result_item = {
                "id": item_id,
                "source": "gsm8k",
                "split": split,
                "original_question": question,
                "original_cot": answer_text,
                "modern_question": parsed.get("vernacular_question", ""),
                "modern_cot": parsed.get("vernacular_cot", ""),
                "classical_question": parsed.get("classical_question", ""),
                "classical_cot": parsed.get("classical_cot", ""),
                "structured_cot": parsed.get("structured_cot", ""),
                "answer": final_answer or "",
            }
            results.append(result_item)
            processed_ids.add(item_id)
            success += 1

            # 写入断点文件
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            print(f"  ✓ 成功 (白话文题目:{len(result_item['modern_question'])}字, "
                  f"白话文CoT:{len(result_item['modern_cot'])}字, "
                  f"文言文题目:{len(result_item['classical_question'])}字, "
                  f"文言文CoT:{len(result_item['classical_cot'])}字)")

        except Exception as e:
            fail += 1
            print(f"  ✗ 失败: {e}")
            # 记录失败项
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(f"{item_id}\t{str(e)[:200]}\n")

        time.sleep(API_CALL_INTERVAL)

    print(f"\n[GSM8K] 处理完成: 成功 {success}, 失败 {fail}, 总计 {total}")
    return results, success, fail


# ============================================================================
# Math23k 处理
# ============================================================================

def process_math23k(client: OpenAI, data: list, checkpoint_path: str,
                    failed_path: str, limit: Optional[int] = None,
                    resume: bool = True) -> tuple:
    """处理 Math23k 数据集，返回 (结果列表, 成功数, 失败数, 不匹配ID列表)。"""
    results = []
    processed_ids = set()
    mismatch_ids = []

    # 断点续传
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    processed_ids.add(obj["id"])
                    results.append(obj)
                    if obj.get("answer_mismatch"):
                        mismatch_ids.append(obj["id"])
        print(f"[Math23k] 从断点恢复，已处理 {len(processed_ids)} 条")

    total = len(data)
    success = len(processed_ids)
    fail = 0

    for item in data:
        raw_id = item["id"]
        item_id = item["uid"]  # 使用带前缀的唯一 ID

        if item_id in processed_ids:
            continue

        if limit is not None and success >= limit:
            break

        original_text = item["original_text"]
        equation = item["equation"]
        ground_truth_ans = item["ans"]

        user_prompt = build_math23k_user_prompt(original_text, equation)

        print(f"\n[Math23k] 处理第 {success+1}/{min(total, limit or total)} 条 "
              f"(raw_id={raw_id}, uid={item_id})...")

        try:
            response = call_api(client, MATH23K_SYSTEM_PROMPT, user_prompt)
            parsed = parse_response(response, MATH23K_MARKER_ORDER)

            vernacular_cot = parsed.get("vernacular_cot", "")
            classical_question = parsed.get("classical_question", "")
            classical_cot = parsed.get("classical_cot", "")

            # 从白话文CoT中提取答案
            extracted_ans = extract_boxed_answer(vernacular_cot)
            if extracted_ans is None:
                extracted_ans = extract_boxed_answer(classical_cot)

            # 比对答案
            answer_mismatch = False
            if extracted_ans is not None:
                norm_extracted = normalize_answer(extracted_ans)
                norm_gt = normalize_answer(ground_truth_ans)
                if norm_extracted != norm_gt:
                    answer_mismatch = True
                    mismatch_ids.append(item_id)
                    print(f"  ⚠ 答案不匹配! 提取: '{extracted_ans}' → "
                          f"标准化: '{norm_extracted}', "
                          f"原始: '{ground_truth_ans}' → 标准化: '{norm_gt}'")
            else:
                answer_mismatch = True
                mismatch_ids.append(item_id)
                print(f"  ⚠ 未能从API响应中提取到 \\boxed{{}} 答案!")

            result_item = {
                "id": item_id,
                "source": "math23k",
                "split": item.get("split", ""),
                "raw_id": raw_id,
                "original_text": original_text,
                "equation": equation,
                "ground_truth_ans": ground_truth_ans,
                "vernacular_question": original_text,  # 原题已是中文白话文
                "vernacular_cot": vernacular_cot,
                "classical_question": classical_question,
                "classical_cot": classical_cot,
                "extracted_ans": extracted_ans or "",
                "answer_mismatch": answer_mismatch,
                "answer": ground_truth_ans,
            }
            results.append(result_item)
            processed_ids.add(item_id)
            success += 1

            # 写入断点
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            status = "⚠答案不一致" if answer_mismatch else "✓"
            print(f"  {status} 成功 (白话文CoT:{len(vernacular_cot)}字, "
                  f"文言文题目:{len(classical_question)}字, "
                  f"文言文CoT:{len(classical_cot)}字, "
                  f"提取答案:{extracted_ans})")

        except Exception as e:
            fail += 1
            print(f"  ✗ 失败: {e}")
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(f"{item_id}\t{str(e)[:200]}\n")

        time.sleep(API_CALL_INTERVAL)

    print(f"\n[Math23k] 处理完成: 成功 {success}, 失败 {fail}, 总计 {total}")
    print(f"[Math23k] 答案不匹配数: {len(mismatch_ids)}")

    return results, success, fail, mismatch_ids


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="数学应用题数据集构建工具 - 使用 DeepSeek API 生成中文/文言文版本"
    )
    parser.add_argument("--source", type=str, default="all",
                        choices=["gsm8k", "math23k", "all"],
                        help="要处理的数据源 (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="指定 train 的处理条数；test 会根据 --ratio 自动计算")
    parser.add_argument("--ratio", type=str, default="8:2",
                        help="train:test 比例，用于根据 --limit 计算 test 数量 (default: 8:2)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        choices=["train", "test"],
                        help="要处理的 split (default: train test)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="输出根目录；不同数据集会写入该目录下的同名子文件夹")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印配置信息，不实际调用 API")
    parser.add_argument("--no-resume", action="store_true",
                        help="不从断点恢复，重新处理所有数据")
    args = parser.parse_args()

    # 显示配置
    print("=" * 60)
    print("  数学应用题数据集构建工具")
    print("=" * 60)
    print(f"  API: DeepSeek ({MODEL_NAME})")
    print(f"  API Key: {'已设置' if DEEPSEEK_API_KEY else '❌ 未设置!'}")
    print(f"  数据源: {args.source}")
    print(f"  Split: {', '.join(args.splits)}")
    print(f"  Train 限制: {args.limit if args.limit else '无限制'}")
    print(f"  Train:Test 比例: {args.ratio}")
    print(f"  断点续传: {'否' if args.no_resume else '是'}")
    print(f"  输出目录: {args.output_dir}")
    print("=" * 60)

    try:
        parse_ratio(args.ratio)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

    if args.dry_run:
        print("\n[Dry-run 模式] 仅检查数据，不调用 API。")
        if args.source in ("gsm8k", "all"):
            for split in args.splits:
                try:
                    df = load_gsm8k(split)
                    planned = split_limit_for(split, args.limit, args.ratio)
                    print(f"  GSM8K {split}: 原始 {len(df)} 条, 计划处理 {format_limit(planned)} 条")
                except FileNotFoundError as e:
                    print(f"  GSM8K {split}: 文件不存在 ({e})")
                except ImportError as e:
                    planned = split_limit_for(split, args.limit, args.ratio)
                    print(f"  GSM8K {split}: parquet 依赖缺失，无法读取条数；计划处理 {format_limit(planned)} 条")
                    print(f"    {e}")
        if args.source in ("math23k", "all"):
            for split in args.splits:
                data = load_math23k(split)
                planned = split_limit_for(split, args.limit, args.ratio)
                print(f"  Math23k {split}: 原始 {len(data)} 条, 计划处理 {format_limit(planned)} 条")
        return

    if not DEEPSEEK_API_KEY:
        print("\n❌ 错误: 未设置 DEEPSEEK_API_KEY 环境变量。")
        print("请设置: set DEEPSEEK_API_KEY=你的密钥  (Windows)")
        print("或:     export DEEPSEEK_API_KEY=你的密钥  (Linux/Mac)")
        sys.exit(1)

    client = create_client()

    all_results = []
    resume = not args.no_resume

    # ------- 处理 GSM8K -------
    if args.source in ("gsm8k", "all"):
        print("\n" + "=" * 60)
        print("  处理 GSM8K 数据集")
        print("=" * 60)
        gsm8k_results = []
        gsm8k_split_summaries = []
        gsm8k_dataset_dir = os.path.join(args.output_dir, "gsm8k")
        for split in args.splits:
            paths = dataset_output_paths(args.output_dir, "gsm8k", split)
            os.makedirs(paths["dataset_dir"], exist_ok=True)

            df = load_gsm8k(split)
            print(f"\n加载 GSM8K {split}: {len(df)} 条")
            split_limit = split_limit_for(split, args.limit, args.ratio)
            print(f"  计划处理: {format_limit(split_limit)} 条")

            results_gsm8k, succ, fail = process_gsm8k(
                client,
                df,
                paths["checkpoint"],
                paths["failed"],
                split,
                limit=split_limit,
                resume=resume,
            )
            print(f"  写入 GSM8K {split} 数据集: {paths['split_file']}")
            with open(paths["split_file"], "w", encoding="utf-8") as f:
                for item in results_gsm8k:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            all_results.extend(results_gsm8k)
            gsm8k_results.extend(results_gsm8k)
            gsm8k_split_summaries.append(
                {
                    "split": split,
                    "raw_items": len(df),
                    "planned_limit": split_limit,
                    "success": succ,
                    "failed": fail,
                    "records_written": len(results_gsm8k),
                    "dataset_file": paths["split_file"],
                    "checkpoint_file": paths["checkpoint"],
                    "failed_file": paths["failed"],
                }
            )
        write_dataset_metadata(gsm8k_dataset_dir, "gsm8k", args, gsm8k_split_summaries)

    # ------- 处理 Math23k -------
    if args.source in ("math23k", "all"):
        print("\n" + "=" * 60)
        print("  处理 Math23k 数据集")
        print("=" * 60)

        math23k_results = []
        math23k_mismatch_ids = []
        math23k_split_summaries = []
        math23k_dataset_dir = os.path.join(args.output_dir, "math23k")
        for split in args.splits:
            paths = dataset_output_paths(args.output_dir, "math23k", split)
            os.makedirs(paths["dataset_dir"], exist_ok=True)

            data = load_math23k(split)
            print(f"加载 Math23k {split}: {len(data)} 条")
            split_limit = split_limit_for(split, args.limit, args.ratio)
            print(f"  计划处理: {format_limit(split_limit)} 条")

            results_math23k, succ, fail, mismatch_ids = process_math23k(
                client,
                data,
                paths["checkpoint"],
                paths["failed"],
                limit=split_limit,
                resume=resume,
            )
            print(f"  写入 Math23k {split} 数据集: {paths['split_file']}")
            with open(paths["split_file"], "w", encoding="utf-8") as f:
                for item in results_math23k:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            all_results.extend(results_math23k)
            math23k_results.extend(results_math23k)
            math23k_mismatch_ids.extend(mismatch_ids)
            math23k_split_summaries.append(
                {
                    "split": split,
                    "raw_items": len(data),
                    "planned_limit": split_limit,
                    "success": succ,
                    "failed": fail,
                    "records_written": len(results_math23k),
                    "answer_mismatch": len(mismatch_ids),
                    "dataset_file": paths["split_file"],
                    "checkpoint_file": paths["checkpoint"],
                    "failed_file": paths["failed"],
                }
            )
        write_dataset_metadata(math23k_dataset_dir, "math23k", args, math23k_split_summaries)

        # 输出不匹配 ID 汇总
        if math23k_mismatch_ids:
            print("\n" + "!" * 60)
            print(f"⚠ 警告: Math23k 中有 {len(math23k_mismatch_ids)} 条数据答案不匹配!")
            print("不匹配 ID 列表:")
            for mid in math23k_mismatch_ids:
                print(f"  - {mid}")
            print("!" * 60)

    # ------- 汇总所有不匹配 ID -------
    all_mismatch = []
    for item in all_results:
        if item.get("answer_mismatch"):
            all_mismatch.append(item["id"])

    if all_mismatch:
        mismatch_log = os.path.join(args.output_dir, "math23k", "mismatch_log.txt")
        os.makedirs(os.path.dirname(mismatch_log), exist_ok=True)
        print(f"\n写入答案不匹配日志: {mismatch_log}")
        with open(mismatch_log, "w", encoding="utf-8") as f:
            f.write(f"答案不匹配记录 (共 {len(all_mismatch)} 条)\n")
            f.write("=" * 60 + "\n")
            for mid in all_mismatch:
                # 找到对应条目获取更多信息
                matched = [x for x in all_results if x["id"] == mid]
                if matched:
                    m = matched[0]
                    f.write(f"id: {mid}")
                    if m.get("extracted_ans"):
                        f.write(f"  |  提取答案: {m['extracted_ans']}")
                    if m.get("answer"):
                        f.write(f"  |  原始答案: {m['answer']}")
                    f.write("\n")
                else:
                    f.write(f"id: {mid}\n")

    print("\n✓ 全部处理完成!")


if __name__ == "__main__":
    main()
