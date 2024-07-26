import os
import argparse
import torch
import numpy as np
import random
import time
from flint.utils.logger import Logger
from typing import Optional
import sys

from dataclasses import dataclass, field
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', type=str, default='None', choices=['cola', 'sst2', 'mrpc', 'stsb',
                                                                     'qqp', 'mnli','qnli','rte'], help='random seed')
    # common param
    parser.add_argument('--model_zoo', type=str, default='None', choices=['None', 'mmdet', 'mmcls', 'mmseg'], help='random seed')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--logger', action='store_true', help='log')

    # data param
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'ImageNet'
                        ,'mmdet','mmcls','mmseg'], help='dataset name (default: CIFAR10)')
    parser.add_argument('--data_path', type=str, default='data/', help='path for store traning data')
    parser.add_argument('--test_path', type=str, default='data/', help='path for store testing data')

    # net param
    parser.add_argument('--ckpt', type=str, default='', help='path for teacher model')
    parser.add_argument('--net', type=str, default='', help='path for teacher model')
    parser.add_argument('--skip_fptest', action='store_true', help='skip float model acc test')

    # run param
    parser.add_argument('--test_mode', type=str, default='acc', choices=['acc', 'cos', 'both'], help='test mode')
    parser.add_argument('--batch_size', type=int, default='32', help='batchsize')
    parser.add_argument('--fold_bn', action='store_true', help='fold bn layers')
    
    #  quant param
    parser.add_argument('--wbit', type=int, default='4', help='bit width for weight')
    parser.add_argument('--abit', type=int, default='4', help='bit width for activation')
    parser.add_argument('--wfmt', type=str, default='int', help='format for weight')
    parser.add_argument('--afmt', type=str, default='int', help='format for activation')
    parser.add_argument('--enable_int', action='store_true', help='mo4w')
    
    parser.add_argument('--quant_pos', type=str, default='out', choices=['out','in'], help='quant function')
    parser.add_argument('--quant_all', action='store_true', help='quant all layers')
    
    parser.add_argument('--mo4w', action='store_true', help='measure out for weight quant')
    parser.add_argument('--limited', action='store_true', help='measure out for weight quant')
    parser.add_argument('--fast', action='store_true', help='measure out for weight quant')
    
    # cali param
    parser.add_argument('--cali_num', type=int, default='200', help='data num for calibration')
    parser.add_argument('--cali_mode', type=str, default='global', choices=['global','batch'], help='quant function')
    parser.add_argument('--w_scale_method', type=str, default='max', choices=['max','mse','update_max'], help='scale method')
    parser.add_argument('--a_scale_method', type=str, default='mse', choices=['max','meanmax','mse','ema'], help='scale method')
    parser.add_argument('--wnorm', type=float, default='2', help='norm for activation')
    parser.add_argument('--anorm', type=float, default='2', help='norm for activation')
    
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    
if __name__ == '__main__':
    
    global args
    args = parse()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    args.net = args.task
    os.makedirs('log', exist_ok=True)
    args.model_label = '{}-{}'.format(args.net, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    print(args.model_label)

    logger = Logger('log/{}.txt'.format(args.model_label), state=args.logger)
    logger.write('config', args)
    args.logger = logger
    
    # args limit
    if args.mo4w:
        logger.info("Use no4w mode")
        args.quant_pos = "in"
    else:
        logger.info("Not use no4w mode")
    if args.limited:
        assert args.mo4w 
    # due to torch.matmul() call not only once
    assert args.cali_mode == 'batch'
 
    from test_glue import run_train

    setup_seed(args.seed)
    task_name = args.task
    sys.argv = ['main_glue.py', '--config_name', 'bert-base-uncased', '--tokenizer_name', 'bert-base-uncased',
                '--model_name_or_path', args.ckpt,
                '--task_name', task_name, '--max_seq_length', '128',
                '--do_eval', '--per_device_train_batch_size','32','--output_dir','log_glue']
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    run_train(args, model_args, data_args, training_args)
    
    args.logger.close()
