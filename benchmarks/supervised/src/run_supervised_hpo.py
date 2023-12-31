import logging
import random
import warnings
import os
import typing
import sys
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining

import evaluate

import datasets
from datasets import Value, load_dataset
from datasets import load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

LABEL_LIST = [
    "Aggregate",
    "Has Human Subjects",
    "Has Target Disease",
    "Cohort Study or Clinical Trial",
    "Has Quantitative Outcome Measure",
    "Has Study Drug(s)",
    "Has Population Size",
    "Has Comparator Group"
]

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

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
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class TuningArguments:
    """
    Arguments for raytune population based training.
    """

    num_trials: int = field(
        default=10, metadata={"help": "Number of tune trials to run."}
    )
    cpus_per_trial: int = field(
        default=None, metadata={"help": "Number of cpus to allocate to each trial."}
    )
    gpus_per_trial: Optional[int] = field(
        default=None, metadata={"help": "Number of GPUs to allocate to each trial."}
    )
    smoke_test: bool = field(
        default=False,
        metadata={"help": "Whether this run is a smoke test or not."},
    )
    time_attr: str = field(
        default="training_iteration",
        metadata={"help": "Time attribute for perturbation interval population based training."},
    )
    perturbation_interval: int = field(
        default=1, metadata={"help": "Perturbation interval for population based training."}
    )
    tune_metric: str = field(
        default="eval_loss",
        metadata={"help": "Metric to use for optimization."},
    )
    mode: str = field(
        default="min",
        metadata={"help": "Whether to maximize or minimize tune_metric."},
    )
    max_weight_decay: Optional[float] = field(
        default=None,
        metadata={"help": "Upper bound on uniform distribution for weight decay."},
    )
    max_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "Upper bound on uniform distribution for learning rate."},
    )
    tuning_train_epochs: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of number of training epochs from which to choose."},
    )
    tuning_batch_sizes: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Per device train batch sizes for tuning."},
    )

def main():
    """Main driver
    Parse args into dataclasses and set up logging,
    then hand off to train for hpo 
    """


    # Parse command line arguments
    hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TuningArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, tuning_args, training_args = hf_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, tuning_args, training_args = hf_parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Tuning parameters {tuning_args}")

    train(model_args, data_args, tuning_args, training_args)


def train(model_args: dataclass, data_args: dataclass, tuning_args: dataclass, training_args: dataclass):

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        if os.path.isdir(data_args.dataset_name):
            # Use load_from_disk if local
            raw_datasets = load_from_disk(data_args.dataset_name)

        else:
            # Else load from hub
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir
            )
            # Try print some info about the dataset
            logger.info(f"Dataset loaded: {raw_datasets}")
            logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir
            )


    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    is_multi_label = False

    if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
        is_multi_label = True
        logger.info("Label type is list, doing multi-label classification")

    label_list = LABEL_LIST

    label_list.sort()
    num_labels = len(label_list)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )

    if is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    if training_args.do_train: 
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    else:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(config.label2id))
        label_to_id = config.label2id


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        return model

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if "label" in examples:
            result["label"] = examples["label"]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric_name is not None:
        metric = (
            evaluate.load(data_args.metric_name, config_name="multilabel")
            if is_multi_label
            else evaluate.load(data_args.metric_name)
        )
        logger.info(f"Using metric {data_args.metric_name} for evaluation.")
    else:
        if is_multi_label:
            f1_metric = evaluate.load("f1", config_name="multilabel")
            prec_metric = evaluate.load("precision", config_name="multilabel")
            rec_metric = evaluate.load("recall", config_name="multilabel")
            logger.info(
                "Using multilabel F1, precision and recall for multi-label classification task, you can use --metric_name to overwrite."
            )
        else:
            metric = evaluate.load("accuracy")
            logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        if is_multi_label:
            preds = np.array([np.where(p > 0.5, 1, 0) for p in preds])
            result = {}
            result["f1_micro"] = f1_metric.compute(predictions=preds,
                                                   references=p.label_ids,
                                                   average="micro")["f1"]
            result["f1_macro"] = f1_metric.compute(predictions=preds,
                                                   references=p.label_ids,
                                                   average="macro")["f1"]
            result["precision_micro"] = prec_metric.compute(predictions=preds,
                                                            references=p.label_ids,
                                                            average="micro")["precision"]
            result["precision_macro"] = prec_metric.compute(predictions=preds,
                                                            references=p.label_ids,
                                                            average="macro")["precision"]
            result["recall_micro"] = rec_metric.compute(predictions=preds, 
                                                        references=p.label_ids, 
                                                        average="micro")["recall"]
            result["recall_macro"] = rec_metric.compute(predictions=preds, 
                                                        references=p.label_ids, 
                                                        average="macro")["recall"]

        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
                
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is 
    # passed to Trainer, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # FIXME: This may be deprecated. Throwing ds.count() error. 
    # Apparently ray datasets arent written in torch under the hood

    # if training_args.do_train:
    #     train_dataset = ray.data.from_huggingface(train_dataset)
    # if training_args.do_eval:
    #     eval_dataset = ray.data.from_huggingface(eval_dataset)
    # if training_args.do_predict:
    #     predict_dataset = ray.data.from_huggingface(predict_dataset)        

    # Initialize our Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    tune_config = {
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "num_train_epochs": (
            tune.choice(tuning_args.tuning_train_epochs)
            if tuning_args.tuning_train_epochs is not None
            else training_args.num_train_epochs
        ),
        "max_steps": 1 if tuning_args.smoke_test else -1  # Used for smoke test.
    }

    hyperparameter_mutations = {}
    if tuning_args.max_weight_decay is not None:
        hyperparameter_mutations["weight_decay"] = (
            tune.uniform(training_args.weight_decay, tuning_args.max_weight_decay)
        )
    if tuning_args.max_learning_rate is not None:
        hyperparameter_mutations["learning_rate"] = (
            tune.uniform(training_args.learning_rate, tuning_args.max_learning_rate)
        )
    if tuning_args.tuning_batch_sizes is not None:
        hyperparameter_mutations["per_device_train_batch_size"] = (
            tuning_args.tuning_batch_sizes
        )

    scheduler = PopulationBasedTraining(
        time_attr=tuning_args.time_attr,
        metric=tuning_args.tune_metric,
        mode=tuning_args.mode,
        perturbation_interval=tuning_args.perturbation_interval,
        hyperparam_mutations=hyperparameter_mutations
    )

    if is_multi_label:
        metric_columns=["eval_loss", "eval_f1_micro", "eval_prec_micro", 
                        "eval_rec_micro", "epoch", "training_iteration"]
    else:
        metric_columns=["eval_loss", "eval_accuracy", "epoch", "training_iteration"]

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/device",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=metric_columns
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=tuning_args.num_trials,
        resources_per_trial={"cpu": tuning_args.cpus_per_trial,
                             "gpu": tuning_args.gpus_per_trial},
        scheduler=scheduler,
        checkpoint_score_attr=tuning_args.time_attr,
        stop={"training_iteration": 1} if tuning_args.smoke_test else None,
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        # local_dir=training_args.output_dir,
        name=model_args.model_name_or_path.replace("/", "_") + "_supervised_pbt",
        log_to_file=True
    )

if __name__ == "__main__":
    main()

