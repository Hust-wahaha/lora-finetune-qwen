from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, get_peft_model
from swift import get_model_processor, get_template
from swift.dataset import EncodePreprocessor
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.utils import find_all_linears, get_logger, get_model_parameter_info, seed_everything

logger = get_logger()
seed_everything(42)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET_PATH = ROOT / 'data' / 'final' / 'train_s800.jsonl'
VAL_DATASET_PATH = ROOT / 'data' / 'final' / 'val_s800.jsonl'
RUNS = ROOT / 'runs'
MODEL_ID = 'Qwen/Qwen3.5-0.8B'
SYSTEM = '你是一个擅长中文数学应用题的助手，要求输出简洁、准确、可验证。'


def make_run_dir() -> Path:
    run_dir = RUNS / datetime.now().strftime('%Y%m%d_%H%M%S')
    for sub in ['logs', 'checkpoints', 'predictions', 'metrics', 'samples']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    run_dir = make_run_dir()
    output_dir = run_dir / 'checkpoints'
    with (run_dir / 'metrics' / 'run_config.json').open('w', encoding='utf-8') as f:
        json.dump({'model': MODEL_ID, 'train_dataset': str(TRAIN_DATASET_PATH), 'val_dataset': str(VAL_DATASET_PATH), 'run_dir': str(run_dir)}, f, ensure_ascii=False, indent=2)

    model, tokenizer = get_model_processor(MODEL_ID)
    template = get_template(tokenizer, default_system=SYSTEM, max_length=1024)
    template.set_mode('train')

    target_modules = find_all_linears(model)
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=32, target_modules=target_modules))
    logger.info(get_model_parameter_info(model))

    train_dataset = hf_load_dataset('json', data_files=str(TRAIN_DATASET_PATH), split='train')
    val_dataset = hf_load_dataset('json', data_files=str(VAL_DATASET_PATH), split='train')
    train_dataset = EncodePreprocessor(template=template)(train_dataset)
    val_dataset = EncodePreprocessor(template=template)(val_dataset)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=25,
        eval_strategy='steps',
        eval_steps=25,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=42,
    )

    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(model=model, args=args, template=template, train_dataset=train_dataset, eval_dataset=val_dataset)
    trainer.train()

    with (run_dir / 'metrics' / 'last_checkpoint.txt').open('w', encoding='utf-8') as f:
        f.write(str(trainer.state.last_model_checkpoint or output_dir))


if __name__ == '__main__':
    main()
