import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Union

import datasets
import torch
import transformers
from arguments import DataTrainingArguments, ModelArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
datasets.logging.set_verbosity_error()


@dataclass
class CustomModelArguments(ModelArguments):
    model_name_or_path: Optional[str] = field(default="Qwen2.5-7B-Instruct")


@dataclass
class CustomDataArguments(DataTrainingArguments):
    dataset_name: str = field(default="sky_original_dataset.json", metadata={"help": "数据集路径"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "可以设置很小的样本数用于快速调试，默认是None，即使用全部数据集。原始分为训练样本数和验证样本数，这里只设置训练样本数，验证样本数会自动从训练样本数中划分一部分作为验证集"})
    validation_split_percentage: Optional[int] = field(default=2, metadata={"help": "验证集样本数占总训练集样本数的比例，单位为%"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "数据预处理的工作进程数量"})
    cutoff_len: int = field(default=4096, metadata={"help": "输入输出拼接到一起之后的最大长度"})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # 这里重写一些参数，更多参数可进入父类TrainingArguments查看
    output_dir: str = field(default="save/cloud_sft1_full", metadata={"help": "模型权重保存路径"})

    per_device_train_batch_size: int = field(default=1, metadata={"help": "每个设备的训练batch_size，注意多卡训练总batch_size要乘以卡数"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "每个设备的验证batch_size，注意多卡验证总batch_size要乘以进程数"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "梯度累积步数"})

    # dataclass的默认值不能输入可变类型，如列表这种会在多个实例中共享，所以需要使用default_factory
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"], metadata={"help": "标签名称，不设置会弹警告，但也不影响训练"})
    learning_rate: float = field(default=5e-5, metadata={"help": "初始学习率，后续根据学习率策略会变化"})
    num_train_epochs: float = field(default=3.0, metadata={"help": "总训练轮数"})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "学习率调度策略"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "线性预热比例"})
    bf16: bool = field(default=True, metadata={"help": "是否使用bf16精度，如果为False，则使用fp16精度"})
    # gradient_checkpointing: bool = field(default=True, metadata={"help": "是否启用梯度检查点，默认False"})

    logging_steps: float = field(default=25, metadata={"help": "打印日志间隔步数，如果设置小于1的float值，则按epoch比例记录"})
    save_steps: float = field(default=250, metadata={"help": "保存模型间隔步数，如果设置小于1的float值，则按epoch比例保存"})
    save_only_model: bool = field(default=True, metadata={"help": "是否只保存模型，如果为False，则保存优化器、调度器和rng状态"})
    eval_strategy: str = field(default="steps", metadata={"help": "评估策略，可选[no, steps, epoch]"})
    eval_steps: int = field(default=250, metadata={"help": "评估间隔步数，如果设置小于1的float值，则按epoch比例评估"})

    report_to: str = field(default="wandb", metadata={"help": "报告结果和日志的集成列表，支持多种集成，如wandb、tensorboard等，none表示不报告，all表示报告所有已安装的集成"})
    run_name: Optional[str] = field(default="naive_cloud_sft_full", metadata={"help": "wandb运行任务名称，若为None，则默认使用output_dir的名字"})

    # DDP分布式训练参数，若单卡训练需注释掉
    # ddp_backend: str = field(default="nccl", metadata={"help": "分布式训练后端，可选[nccl, gloo]"})
    # ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "是否在分布式训练中查找未使用的参数"})
    # dataloader_num_workers: int = field(default=4, metadata={"help": "数据加载器的工作进程数量"})

    # deepspeed训练方式，根据不同的zero stage选择不同的配置文件
    deepspeed: Optional[Union[dict, str]] = field(default='deepspeed/ds_z3_config.json', metadata={"help": "deepspeed配置文件路径，或已加载json的字典"})


def main():
    parser = HfArgumentParser((CustomModelArguments, CustomDataArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果只传递一个参数给脚本并且它是json文件的路径,1
        # 让我们解析它以获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # 默认情况下，training_args.log_level 是 WARNING 模式，这里设置为 INFO 级别打印更多信息，后续log设置基于这个值
        transformers.utils.logging.set_verbosity_info()

    # 获取当前的日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)  # 统一设置所有日志级别
    transformers.utils.logging.set_verbosity(log_level)  # 设置 transformers 库的日志级别
    transformers.utils.logging.enable_default_handler()  # 启用默认的日志处理器，确保日志能被正确输出
    transformers.utils.logging.enable_explicit_format()  # 启用更明确的日志格式，让日志输出更易读

    # 将datasets的日志级别设置为WARNING，否则在处理数据时打印太多信息
    logging.getLogger("datasets").setLevel(logging.WARNING)

    # 打印training_args参数，不想看到可以注释
    logger.info("Training/evaluation parameters:")
    for key, value in training_args.__dict__.items():
        logger.info(f"  {key} = {value}")

    # 在初始化模型之前设置种子
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # qwen2默认使用的是sdpa，为加速计算以及节省显存，可开启flash_attention_2
    # flash_attention安装容易出现版本兼容问题，可前往 https://github.com/Dao-AILab/flash-attention/releases/ 针对 python+pytorch+cuda 版本选择 whl 文件下载安装
    config._attn_implementation = 'flash_attention_2'
    # config._attn_implementation = 'eager'
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left')
    # 加载模型，注意设置torch_dtype='auto'，就会自动加载预训练模型的数据类型，不用后面再model.half()
    # 不要设置device_map='auto'，否则会进行模型并行，将不同层切分到不同的设备上
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                 config=config, 
                                                 torch_dtype='auto')

    # 使用LoRA训练参数
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,  # 低秩矩阵的秩
    #     lora_alpha=16,  # 缩放因子，一般为秩的两倍
    #     lora_dropout=0.1,  # 随机失活，一般为0.1
    #     target_modules='all-linear'  # 需要添加LoRA的模块，在训练新的模型时需注意，例如模型后面接了新的线性层
    # )
    # # 创建lora结构
    # model = get_peft_model(model, peft_config)
    # # 打印可训练参数
    # model.print_trainable_parameters()
    # 设置模型为训练模式，否则默认使用推理模式，很快就梯度消失
    model.train()

    # 数据处理，提前将数据转换为模型输入的形式，非dataloader阶段的处理
    # 处理方式较灵活，可根据自己需求修改
    def tokenize_function(examples):
        # examples是datasets的格式，是一个字典，key是数据集中的字段名，value是数据集中的数据
        model_inputs = defaultdict(list)
        bos_token_id = tokenizer.encode("<|im_start|>")[0]  # 获取bos_token_id
        for i in range(len(examples[text_column_name[1]])):  # 循环处理每个样本
            # 构造和推理时相同的结构，若instruction为空，则系统提示词可默认设置 "You are a helpful assistant"
            message = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": examples[text_column_name[1]][i]},  # input
                {"role": "assistant", "content": examples[text_column_name[2]][i]}  # output
            ]
            # 尽量使用模型自带的模板，不要输入输出直接拼接，与推理时保持一致
            format_inputs = tokenizer.apply_chat_template(message, tokenize=True)
            # 某几个长度过长的样本会占用太多显存导致OOM，这里进行过滤
            # 若直接截断，输入输出生成一般貌似也不太合理，这里直接进行过滤，可根据需求修改
            if len(format_inputs) > data_args.cutoff_len:
                continue
            # 获取bos_token_id的位置，用于后面构造label
            last_bos_index = torch.where(torch.tensor(format_inputs) == bos_token_id)[0][-1].tolist()
            # if last_bos_index > data_args.cutoff_len-100:
            #     format_inputs = format_inputs[last_bos_index - data_args.cutoff_len+100:]
            #     format_inputs = format_inputs[:data_args.cutoff_len]
            #     last_bos_index = torch.where(torch.tensor(format_inputs) == bos_token_id)[0][-1].tolist()
            # 构造label，-100表示不计算梯度，只计算input_ids的梯度
            label_ids = [-100] * (last_bos_index + 1) + format_inputs[last_bos_index + 1:]

            # format_inputs = format_inputs + [tokenizer.eos_token_id] * (data_args.cutoff_len - len(format_inputs))
            # label_ids = label_ids + [-100] * (data_args.cutoff_len - len(label_ids))
            # # 将处理后的数据添加到model_inputs中
            model_inputs["input_ids"].append(format_inputs)
            model_inputs["attention_mask"].append([1] * len(format_inputs))
            model_inputs["labels"].append(label_ids)
        return model_inputs
    
    # dataloader阶段调用的collate_fn函数，将单个数据转化成batch形式，features是一个列表，长度为batch_size，里面的每个元素都是tokenize_function处理的形式
    # 当前操作主要进行不同batch的对齐，以及attention_mask和labels的填充
    def collate_fn(features):
        batch = {}
        # 获取每个batch中最大的长度
        max_length = max(len(f["input_ids"]) for f in features)

        # 对齐不同batch的长度【右填充】
        new_input_ids = [f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"])) for f in features]
        new_attention_mask = [f["attention_mask"] + [0] * (max_length - len(f["attention_mask"])) for f in features]
        new_labels = [f["labels"] + [-100] * (max_length - len(f["labels"])) for f in features]

        # 对齐不同batch的长度【左填充】
        new_input_ids = [[tokenizer.pad_token_id] * (max_length - len(f["input_ids"])) + f["input_ids"] for f in features]
        new_attention_mask = [[0] * (max_length - len(f["attention_mask"])) + f["attention_mask"] for f in features]
        new_labels = [[-100] * (max_length - len(f["labels"])) + f["labels"] for f in features]

        # 将处理后的数据添加到batch中
        batch["input_ids"] = torch.tensor(new_input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(new_attention_mask, dtype=torch.long)
        batch["labels"] = torch.tensor(new_labels, dtype=torch.long)
        return batch

    # 加载数据集，可以指定本地路径加载，或提供 https://huggingface.co/datasets/ 上公共数据集的名字，此时会联网下载（国内下载经常失败，建议提前下载到本地）
    # 具体用法见：https://huggingface.co/docs/datasets/loading
    with training_args.main_process_first(desc="load dataset"):
        raw_datasets = load_dataset(
            path=data_args.dataset_name.split('.')[-1],  # 数据集格式，例如'csv'、'json'、'txt'等
            data_files=data_args.dataset_name,  # 数据集路径，可以是本地路径，也可以是huggingface上公共数据集的名字
            # split='train',  # 指定数据集的哪个子集，例如'train'、'validation'、'test'等，默认是'train'
            # cache_dir=model_args.cache_dir,  # 缓存目录，用于存储下载的数据集
            # use_auth_token=True if model_args.use_auth_token else None,  # 是否使用认证令牌，例如用于huggingface上的私有数据集
        )
    # 获取数据集中的字段，方便后面取出
    text_column_name = list(raw_datasets["train"].features)  # 这里使用alpaca数据集格式，所以是['instruction', 'input', 'output']

    # 如果设置了max_train_samples或max_eval_samples，则只取部分数据
    if data_args.max_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(min(data_args.max_samples, len(raw_datasets["train"]))))

    # 1. 在分布式训练中,确保主进程(rank 0)首先执行数据处理操作
    # 2. 其他进程会等待主进程完成后再继续执行
    # 3. 这样可以避免多个进程同时处理数据时可能产生的竞争条件和数据不一致问题
    with training_args.main_process_first(desc="pre-process dataset"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,  # 数据处理函数
            batched=True,  # 是否分批处理
            batch_size=1000,  # 分批处理每个batch的大小
            num_proc=16,  # 多进程处理，加快处理速度
            remove_columns=text_column_name,  # 删除原始数据集中的字段，若不删除，tokenize_function处理之后的数据长度必须和原始相同，否则会数量不一致报错
            load_from_cache_file=False,  # 是否从缓存加载，之前的处理都会缓存下来，这里可从上次的缓存加载，无需再次处理。注意若tokenize_function函数修改需重新处理
            # logging_level="ERROR",
        )
    logger.info(f"tokenized_datasets: {tokenized_datasets}")
    dataset_len = []
    for i in range(len(tokenized_datasets["train"])):
        dataset_len.append(len(tokenized_datasets["train"][i]["input_ids"]))
    import numpy as np
    logger.info(f"dataset_count: {len(dataset_len)}")
    logger.info(f"dataset_len: {np.mean(dataset_len)}")

    # 将数据集拆分为训练集和验证集，若原始数据集中没有验证集，则从训练集中划分一部分作为验证集
    if "validation" not in tokenized_datasets:
        # 如果原始数据集中没有验证集，则从训练集中划分一部分作为验证集
        train_test_split = tokenized_datasets["train"].train_test_split(
            test_size=0.02,  # 设置作为验证集的比例
            seed=42,  # 设置随机种子以确保可重复性
        )
        tokenized_datasets["train"], tokenized_datasets["validation"] = train_test_split["train"], train_test_split["test"]
    logger.info(f"训练集样本数: {len(tokenized_datasets['train'])}")
    logger.info(f"验证集样本数: {len(tokenized_datasets['validation'])}")

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    logger.info(f"model注意力计算策略: {model.config._attn_implementation}")
    logger.info(f"model.device: {model.device}")
    # 创建 Trainer 对象
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    train_result = trainer.train()  # 开始训练模型
    trainer.save_model()  # 保存模型权重
    logger.info(f"训练结果: {train_result}")


if __name__ == "__main__":
    main()

'''
朴素版大模型 SFT 微调代码，支持单卡和多卡训练

1. DDP训练方式，在 CustomTrainingArguments 中设置 ddp_backend 参数，并运行以下命令：
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_sft_naive.py

    等同于旧版启动方式：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_sft_naive.py

    DDP分布式训练参考资料：
    https://pytorch.org/tutorials/beginner/dist_overview.html#parallelism-apis


2. deepspeed训练方式，在 CustomTrainingArguments 中设置 deepspeed 参数，并运行以下命令：
deepspeed --num_gpus=8 train_sft_naive.py
'''