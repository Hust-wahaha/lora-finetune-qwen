import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift import get_model_processor, load_dataset, get_template
from swift.dataset import EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from peft import get_peft_model, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

logger = get_logger()
seed_everything(42)

# Hyperparameters for training
# model
model_id_or_path = 'Qwen/Qwen3.5-0.8B'  # model_id or model_path
system = 'You are a helpful assistant.'
output_dir = 'output'

# dataset
dataset = ['AI-ModelScope/alpaca-gpt4-data-zh#500', 'AI-ModelScope/alpaca-gpt4-data-en#500',
           'swift/self-cognition#500']  # dataset_id or dataset_path
data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01  # Split validation set
num_proc = 4  # The number of processes for data loading.
# The following two parameters are used to override the placeholders in the self-cognition dataset.
model_name = ['小黄', 'Xiao Huang']  # The Chinese name and English name of the model
model_author = ['魔搭', 'ModelScope']  # The Chinese name and English name of the model author

# lora
lora_rank = 8
lora_alpha = 32

# training_args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
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
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')

# Obtain the model and template, and add a trainable Lora layer on the model.
model, tokenizer = get_model_processor(model_id_or_path)
logger.info(f'model_info: {model.model_info}')
template = get_template(tokenizer, default_system=system, max_length=max_length)
template.set_mode('train')

target_modules = find_all_linears(model)
lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = get_peft_model(model, lora_config)
logger.info(f'lora_config: {lora_config}')

# Print model structure and trainable parameters.
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')

# Download and load the dataset, split it into a training set and a validation set,
# and encode the text data into tokens.
train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
        model_name=model_name, model_author=model_author, seed=data_seed)

logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# Print a sample
template.print_inputs(train_dataset[0])

# Get the trainer and start the training.
model.enable_input_require_grads()  # Compatible with gradient checkpointing
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    template=template,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()

last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'last_model_checkpoint: {last_model_checkpoint}')