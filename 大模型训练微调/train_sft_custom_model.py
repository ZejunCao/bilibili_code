import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn.functional as F
import transformers
from arguments import DataTrainingArguments, ModelArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import SaveStrategy

logger = logging.getLogger(__name__)
datasets.logging.set_verbosity_error()


@dataclass
class CustomModelArguments(ModelArguments):
    model_name_or_path: Optional[str] = field(default="/data0/zejun7/open_source/bilibili_code/大模型训练微调/model_checkpoint/cloud_sft1_7b_full_0415")


@dataclass
class CustomDataArguments(DataTrainingArguments):
    dataset_name: str = field(default="/data0/zejun7/open_source/bilibili_code/大模型训练微调/data/sky_original_dataset_0415_selfgen_pair.json", metadata={"help": "数据集路径"})
    max_samples: Optional[int] = field(default=None, metadata={"help": "可以设置很小的样本数用于快速调试，默认是None，即使用全部数据集。原始分为训练样本数和验证样本数，这里只设置训练样本数，验证样本数会自动从训练样本数中划分一部分作为验证集"})
    validation_split_percentage: Optional[int] = field(default=2, metadata={"help": "验证集样本数占总训练集样本数的比例，单位为%"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "数据预处理的工作进程数量"})
    cutoff_len: int = field(default=2048, metadata={"help": "输入输出拼接到一起之后的最大长度"})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # 这里重写一些参数，更多参数可进入父类TrainingArguments查看
    output_dir: str = field(default="save/cloud_sft2_full_0414", metadata={"help": "模型权重保存路径"})

    per_device_train_batch_size: int = field(default=1, metadata={"help": "每个设备的训练batch_size，注意多卡训练总batch_size要乘以卡数"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "每个设备的验证batch_size，注意多卡验证总batch_size要乘以进程数"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "梯度累积步数"})

    # dataclass的默认值不能输入可变类型，如列表这种会在多个实例中共享，所以需要使用default_factory
    label_names: Optional[List[str]] = field(default_factory=lambda: ["labels"], metadata={"help": "标签名称，不设置会弹警告，但也不影响训练"})
    learning_rate: float = field(default=1e-5, metadata={"help": "初始学习率，后续根据学习率策略会变化"})
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
    # run_name: Optional[str] = field(default="cloud_sft2_7b_full", metadata={"help": "wandb运行任务名称，若为None，则默认使用output_dir的名字"})

    remove_unused_columns: bool = field(default=False, metadata={"help": "是否在训练过程中删除未使用的列"})
    # DDP分布式训练参数，若单卡训练需注释掉
    # ddp_backend: str = field(default="nccl", metadata={"help": "分布式训练后端，可选[nccl, gloo]"})
    # ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "是否在分布式训练中查找未使用的参数"})
    # dataloader_num_workers: int = field(default=4, metadata={"help": "数据加载器的工作进程数量"})

    # deepspeed训练方式，根据不同的zero stage选择不同的配置文件
    deepspeed: Optional[Union[dict, str]] = field(default='deepspeed/ds_z3_config.json', metadata={"help": "deepspeed配置文件路径，或已加载json的字典"})

class RewardHead(nn.Module):
    def __init__(self, cfg: PretrainedConfig, n_labels: int):
        super().__init__()
        self.reward_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # cfg.hidden_size=3584
        # use same dropout as attention dropout
        self.reward_out_proj = nn.Linear(cfg.hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        hidden_states = self.reward_dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        output = self.reward_out_proj(hidden_states)
        return output


class CloudforQwen2Model(PreTrainedModel):
    base_model_prefix = "model"
    def __init__(self, pretrained_model_name_or_path, config):
        super().__init__(config)
        # self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, config=config)
        self.lm_weight = 1.25
        self.reward_loss = 0
        self.loss = 0
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        # qwen2默认使用的是sdpa，为加速计算以及节省显存，可开启flash_attention_2
        # flash_attention安装容易出现版本兼容问题，可前往 https://github.com/Dao-AILab/flash-attention/releases/ 针对 python+pytorch+cuda 版本选择 whl 文件下载安装
        config._attn_implementation = 'flash_attention_2'
        # config._attn_implementation = 'eager'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side='left')
        # 加载模型，注意设置torch_dtype='auto'，就会自动加载预训练模型的数据类型，不用后面再model.half()
        # 不要设置device_map='auto'，否则会进行模型并行，将不同层切分到不同的设备上
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, 
                                                    config=config, 
                                                    torch_dtype='auto')
        self.reward_head = RewardHead(config, 1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        batch_size, _ = input_ids.shape
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,  # flash attention2 需要 num_items_in_batch 参数，注意引入**kwargs，否则导致loss很大
        )
        hidden_states = output.hidden_states[-1]
        # sequence_lengths = torch.sum(attention_mask, dim=-1) - 1  # 右填充
        # hidden_states_last_token = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        hidden_states_last_token = hidden_states[torch.arange(batch_size, device=hidden_states.device), -1]  # 左填充
        rewards = self.reward_head(hidden_states_last_token)
        chosen_rewards = rewards[:batch_size//2]
        rejected_rewards = rewards[batch_size//2:]
        reward_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()  # Bradley–Terry term
        self.reward_loss += reward_loss.item()
        self.loss += output.loss.item()
        output.loss = reward_loss + self.lm_weight * output.loss
        return output


class CustomTrainer(Trainer):
    # 为了增加reward loss的打印，重写_maybe_log_save_evaluate方法
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["reward_loss"] = round(self.model.reward_loss / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["lm_loss"] = round(self.model.loss / (self.state.global_step - self._globalstep_last_logged), 4)
            self.model.reward_loss = 0
            self.model.loss = 0
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


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
    model = CloudforQwen2Model(model_args.model_name_or_path, config)
    tokenizer = model.tokenizer

    # 使用LoRA训练参数
    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,  # 低秩矩阵的秩
    #     lora_alpha=16,  # 缩放因子，一般为秩的两倍
    #     lora_dropout=0.1,  # 随机失活，一般为0.1
    #     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj', 'lm_head']  # 需要添加LoRA的模块，在训练新的模型时需注意，例如模型后面接了新的线性层
    # )
    # # 创建lora结构
    # model = get_peft_model(model, peft_config)
    # # 打印可训练参数
    # model.print_trainable_parameters()
    # model = model.to("cuda:0")
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
            chosen_message = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": examples[text_column_name[0]][i]},  # input
                {"role": "assistant", "content": examples[text_column_name[2]][i]}  # output
            ]
            rejected_message = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": examples[text_column_name[1]][i]},  # input
                {"role": "assistant", "content": examples[text_column_name[3]][i]}  # output
            ]
            # 尽量使用模型自带的模板，不要输入输出直接拼接，与推理时保持一致
            format_chosen_inputs = tokenizer.apply_chat_template(chosen_message, tokenize=True)
            format_rejected_inputs = tokenizer.apply_chat_template(rejected_message, tokenize=True)
            # 某几个长度过长的样本会占用太多显存导致OOM，这里进行过滤
            # 若直接截断，输入输出生成一般貌似也不太合理，这里直接进行过滤，可根据需求修改
            if len(format_chosen_inputs) > data_args.cutoff_len or len(format_rejected_inputs) > data_args.cutoff_len:
                continue
            # 获取bos_token_id的位置，用于后面构造label
            chosen_last_bos_index = torch.where(torch.tensor(format_chosen_inputs) == bos_token_id)[0][-1].tolist()
            rejected_last_bos_index = torch.where(torch.tensor(format_rejected_inputs) == bos_token_id)[0][-1].tolist()
            chosen_label_ids = [-100] * (chosen_last_bos_index + 1) + format_chosen_inputs[chosen_last_bos_index + 1:]
            rejected_label_ids = [-100] * (rejected_last_bos_index + 1) + format_rejected_inputs[rejected_last_bos_index + 1:]

            # # 将处理后的数据添加到model_inputs中
            model_inputs["chosen_input_ids"].append(format_chosen_inputs)
            model_inputs["rejected_input_ids"].append(format_rejected_inputs)
            model_inputs["chosen_attention_mask"].append([1] * len(format_chosen_inputs))
            model_inputs["rejected_attention_mask"].append([1] * len(format_rejected_inputs))
            model_inputs["chosen_labels"].append(chosen_label_ids)
            model_inputs["rejected_labels"].append(rejected_label_ids)
        return model_inputs
    
    # dataloader阶段调用的collate_fn函数，将单个数据转化成batch形式，features是一个列表，长度为batch_size，里面的每个元素都是tokenize_function处理的形式
    # 当前操作主要进行不同batch的对齐，以及attention_mask和labels的填充
    def collate_fn(features):
        batch = {}
        # 获取每个batch中最大的长度
        input_lens = [len(f["chosen_input_ids"]) for f in features]
        input_lens.extend([len(f["rejected_input_ids"]) for f in features])
        max_length = max(input_lens)

        # 对齐不同batch的长度【左填充】
        new_chosen_input_ids = [[tokenizer.pad_token_id] * (max_length - len(f["chosen_input_ids"])) + f["chosen_input_ids"] for f in features]
        new_rejected_input_ids = [[tokenizer.pad_token_id] * (max_length - len(f["rejected_input_ids"])) + f["rejected_input_ids"] for f in features]
        new_chosen_attention_mask = [[0] * (max_length - len(f["chosen_attention_mask"])) + f["chosen_attention_mask"] for f in features]
        new_rejected_attention_mask = [[0] * (max_length - len(f["rejected_attention_mask"])) + f["rejected_attention_mask"] for f in features]
        new_chosen_labels = [[-100] * (max_length - len(f["chosen_labels"])) + f["chosen_labels"] for f in features]
        new_rejected_labels = [[-100] * (max_length - len(f["rejected_labels"])) + f["rejected_labels"] for f in features]

        final_input_ids = torch.cat([torch.tensor(new_chosen_input_ids, dtype=torch.long), torch.tensor(new_rejected_input_ids, dtype=torch.long)], dim=0)
        final_attention_mask = torch.cat([torch.tensor(new_chosen_attention_mask, dtype=torch.long), torch.tensor(new_rejected_attention_mask, dtype=torch.long)], dim=0)
        final_labels = torch.cat([torch.tensor(new_chosen_labels, dtype=torch.long), torch.tensor(new_rejected_labels, dtype=torch.long)], dim=0)
        # 将处理后的数据添加到batch中
        batch["input_ids"] = final_input_ids
        batch["attention_mask"] = final_attention_mask
        batch["labels"] = final_labels
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
    text_column_name = ['chosen_input', 'rejected_input', 'chosen_critique', 'rejected_critique']  # 这里使用alpaca数据集格式，所以是['instruction', 'input', 'output']

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
    # dataset_len = [len(d["input_ids"]) for d in tokenized_datasets["train"]]
    # import numpy as np
    # logger.info(f"dataset_count: {len(dataset_len)}")
    # logger.info(f"dataset_len: {np.mean(dataset_len)}")
    # logger.info(f"dataset_len_max: {np.max(dataset_len)}")
    # logger.info(f"dataset_len_min: {np.min(dataset_len)}")

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
    # 创建 Trainer 对象
    trainer = CustomTrainer(
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
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_sft_custom_model.py

    等同于旧版启动方式：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_sft_custom_model.py

    DDP分布式训练参考资料：
    https://pytorch.org/tutorials/beginner/dist_overview.html#parallelism-apis


2. deepspeed训练方式，在 CustomTrainingArguments 中设置 deepspeed 参数，并运行以下命令：
deepspeed --num_gpus=4 train_sft_custom_model.py
'''