import json
from pathlib import Path
import os

from peft import PeftModel
from swift.infer_engine import TransformersEngine, RequestConfig, InferRequest
from swift import get_model_processor, get_template

RUN_DIR = Path(os.environ['RUN_DIR'])
CHECKPOINT = os.environ['CHECKPOINT']
MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，请尽量给出简洁且正确的答案。'

QUESTIONS = [
    {'id': 'q1', 'family': 'give_away', 'query': '小明有12颗糖，送给小红5颗，他还剩几颗？', 'gold_answer': '7'},
    {'id': 'q2', 'family': 'box_total', 'query': '一盒铅笔有8支，3盒一共有多少支？', 'gold_answer': '24'},
    {'id': 'q3', 'family': 'equal_division', 'query': '把24个苹果平均分给6个小朋友，每人分几个？', 'gold_answer': '4'},
    {'id': 'q4', 'family': 'unit_price', 'query': '一支钢笔9元，买4支需要多少钱？', 'gold_answer': '36'},
    {'id': 'q5', 'family': 'bundle_count', 'query': '有35朵花，每7朵扎成一束，可以扎几束？', 'gold_answer': '5'},
    {'id': 'q6', 'family': 'weight_remaining', 'query': '妈妈买了2千克苹果，吃掉了300克，还剩多少克？', 'gold_answer': '1700'},
]


def build_engine(adapter=None):
    model, tokenizer = get_model_processor(MODEL_ID)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    template_type = model.model_meta.template
    template = get_template(tokenizer, template_type=template_type, default_system=SYSTEM)
    return TransformersEngine(model, template=template, max_batch_size=6)


def run_one(name, adapter=None):
    engine = build_engine(adapter)
    req_cfg = RequestConfig(max_tokens=512, temperature=0.0, top_p=1.0, top_k=20, repetition_penalty=1.0)
    requests = [InferRequest(messages=[{'role': 'user', 'content': q['query']}]) for q in QUESTIONS]
    resps = engine.infer(requests, req_cfg)
    rows = []
    for q, r in zip(QUESTIONS, resps):
        rows.append({
            'id': q['id'],
            'family': q['family'],
            'query': q['query'],
            'gold_answer': q['gold_answer'],
            'response': r.choices[0].message.content,
        })
    out = RUN_DIR / 'predictions' / f'{name}.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return rows


def main():
    baseline = run_one('baseline', None)
    finetuned = run_one('finetuned', CHECKPOINT)
    summary = {
        'baseline_file': str(RUN_DIR / 'predictions' / 'baseline.json'),
        'finetuned_file': str(RUN_DIR / 'predictions' / 'finetuned.json'),
        'count': len(QUESTIONS),
        'checkpoint': CHECKPOINT,
    }
    (RUN_DIR / 'metrics' / 'compare_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
