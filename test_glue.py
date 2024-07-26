import torch.nn as nn
import torch.nn.functional as F

from utils.metric import test
from flint.quantization import * 
from flint.quantization.quant_func.quant_func import FloatQuantizer
from utils import layer_utils
from utils.layer_utils import *
import copy
from utils.parser import *

from datasets import load_dataset, load_metric
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

def prepare_glue(args, model_args, data_args, training_args):
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1


    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    train_dataset = raw_datasets["train"]
    # if data_args.max_train_samples is not None:
    max_train_samples = min(len(train_dataset), args.cali_num)
    train_dataset = train_dataset.select(range(max_train_samples))
            
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    args.trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    args.eval_dataset = eval_dataset
    args.train_dataset = train_dataset
    args.raw_datasets = raw_datasets
    # args.train_loader = DataLoader(train_dataset, args.batch_size, num_workers=4,shuffle=True)
    # args.test_loader = DataLoader(eval_dataset, 1, num_workers=4,shuffle=False)

def quant(model, args, quant_list, model_path=""):
        # print('-------------',str(type(model)), isinstance(model,Matmul))
        if type(model) in quant_list or 'Matmul' in str(type(model)):
            if isinstance(model, nn.Conv2d):
                quant_mod = QuantizedConv2d(wbits=args.wbit,wfmt=args.wfmt,wnorm=args.wnorm,abits=args.abit,afmt=args.afmt,anorm=args.anorm,
                                cali_num=args.cali_num, w_scale_method=args.w_scale_method, a_scale_method=args.a_scale_method, 
                                quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path, mo4w=args.mo4w,
                                limited=args.limited, fast=args.fast, cali_mode=args.cali_mode)
                quant_mod.set_param(model)
                return quant_mod
            elif isinstance(model, nn.Linear):
                quant_mod = QuantizedLinear(wbits=args.wbit,wfmt=args.wfmt,wnorm=args.wnorm,abits=args.abit,afmt=args.afmt,anorm=args.anorm, 
                                cali_num=args.cali_num, w_scale_method=args.w_scale_method, a_scale_method=args.a_scale_method, 
                                quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path, mo4w=args.mo4w,
                                limited=args.limited, fast=args.fast, cali_mode=args.cali_mode)
                quant_mod.set_param(model)
                return quant_mod
            else:
                quant_mod = QuantLayer(abits=args.abit, afmt=args.afmt, anorm=args.anorm, 
                            layer=model, cali_num=args.cali_num, scale_method=args.a_scale_method, 
                            quant_pos=args.quant_pos, enable_int=args.enable_int, path=model_path,
                            fast=args.fast, cali_mode=args.cali_mode)
                return quant_mod
            
        elif isinstance(model, (nn.Sequential, nn.ModuleList)):
            for n, m in model.named_children():
                new_submodel = quant(m, args, quant_list, model_path+n+'/')
                new_submodel.path = model_path+n+'/'
                setattr(model, n, new_submodel)
            return model
        
        else:
            # print(model)
            for attr in dir(model):
                try:
                    mod = getattr(model, attr)
                except:
                    continue
                if isinstance(mod, nn.Module):  # and 'norm' not in attr:
                    if attr=='base_model':
                        continue
                    new_submodel = quant(mod, args, quant_list, model_path+attr+'/')
                    new_submodel.path = model_path+attr+'/'
                    setattr(model, attr, new_submodel)
            return model

def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        if isinstance(m, (QuantizedConv2d, QuantizedLinear)):
            m.set_quant_state(weight_quant, act_quant)
            
def prepare_net(args,model_args, data_args, training_args):

    Quantlist = [nn.Conv2d, layer_utils.Matmul,
            nn.Linear, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Sigmoid, nn.Softmax, 
            ChannelShuffle, Eltwise, Concat, Slice, Flatten, Adap_avg_pool, Interp, V3Sigmoid]
    
    tasks = [data_args.task_name]
    eval_datasets = [args.eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(args.raw_datasets["validation_mismatched"])
        combined = {}
    
    if not args.skip_fptest:
        args.trainer.evaluate(eval_dataset=args.train_dataset)
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = args.trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            args.trainer.log_metrics("eval-fp", metrics)
            args.trainer.save_metrics("eval-fp", combined if task is not None and "mnli" in task else metrics)
        combined = {}
    
    qnn = args.trainer.model
    qnn = quant(qnn, args, Quantlist)
    
    set_quant_state(qnn, True, True)
    qnn.to('cuda').eval()    
    args.trainer.model = qnn
    args.trainer.evaluate(eval_dataset=args.train_dataset)
    # print(qnn)
    # stop()

    if args.wbit == 8:
        res = {'int8':0, 'e5m2':0, 'e4m3':0,'e3m4':0,'e2m5':0}
    elif args.wbit == 6:
        res = {'int6':0, 'e3m2':0, 'e2m3':0}
    elif args.wbit == 4:
        res = {'int4':0, 'e1m2':0, 'e2m1':0, 'e3m0':0}
    else:
        res = {f'int{args.wbit%32}':0}
    for module in args.trainer.model.modules():
        # print(module)
        if isinstance(module, FloatQuantizer):
            if not module.inited:
                args.logger.info("{} is not inited".format(module.accum_path))
            # print(module, module.numeric)
            num = module.numeric.split()
            for n in num:
                if n in res:
                    res[n] += 1

    args.logger.write('QNN bit dist', res)
        
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = args.trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        args.trainer.log_metrics("eval", metrics)
        args.trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    return 1

    


def run_train(args, model_args, data_args, training_args):
    # parse
    prepare_glue(args, model_args, data_args, training_args)
    assert not(args.cali_num % args.batch_size)
    acc = prepare_net(args, model_args, data_args, training_args)

    return acc
