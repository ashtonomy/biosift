"""
build_dataset
Compile and merge csv's into HF Datasets for binary labels, soft labels and NLI.
"""
import pandas as pd
import os
import logging
from argparse import ArgumentParser
import typing
from datasets import (
    load_dataset, 
    Dataset, 
    DatasetDict,
    Features,
    Sequence,
    Value
)
logger = logging.getLogger(__name__)

DATASET_DIR=os.path.dirname(os.path.dirname(__file__))

LABEL_HYPOTHESES = {
    "Cohort Study or Clinical Trial": {
        "positive": "This study has a cohort study or clinical trial", 
        "negative": "This study does not have any cohorts or clinical trial"
    },
    "Has Comparator Group": {
        "positive": "This study has a control, double-blind, or comparison patient group",
        "negative": "This study does not have any comparison patient group"
    },
    "Has Human Subjects": {
        "positive": "This study has human subjects",
        "negative": "This study does not have human subjects"
    },
    "Has Population Size": {
        "positive": "This study contains population size or sample size information",
        "negative": "This study does not contain population size information"
    },
    "Has Quantitative Outcome Measure": {
        "positive": "This study has quantitative outcomes like numbers, P-value, OR, CI, HR, RR, or patient ratios",
        "negative": "This study does not have any quantitative outcomes"
    },
    "Has Study Drug(s)": {
        "positive": "This study has a target drug",
        "negative": "This study does not have a target drug"
    },
    "Has Target Disease": {
        "positive": "This study has a target disease",
        "negative": "This study does not have a target disease"
    }
}

def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--binary_labels",
        type=str,
        default=os.path.join(DATASET_DIR, "PMID_BinaryLabels.csv")
    )
    arg_parser.add_argument(
        "--soft_labels",
        type=str,
        default=os.path.join(DATASET_DIR, "PMID_SoftLabels.csv")
    )
    arg_parser.add_argument(
        "--abstracts",
        type=str,
        default=os.path.join(DATASET_DIR, "PMID_Title_Abstract.csv")
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(DATASET_DIR, "hf_datasets")
    )
    
    return arg_parser.parse_args()

def convert_to_nli(examples):
    """
    Convert batch of binary labeled examples to nli examples.
    """

    new_examples = {
        "PMID": [],
        "Title": [],
        "Abstract": [],
        "Hypothesis": [],
        "Entailment": []
    }

    for i in range(len(examples["PMID"])):
        for label_name in LABEL_HYPOTHESES.keys():
            for hyp in ["positive", "negative"]:
                new_examples["PMID"].append(examples["PMID"][i])
                new_examples["Title"].append(examples["Title"][i])
                new_examples["Abstract"].append(examples["Abstract"][i])
                new_examples["Hypothesis"].append(LABEL_HYPOTHESES[label_name][hyp])
                
                if hyp == "positive":
                    entailment_label = examples[label_name][i]
                else:
                    entailment_label = int(not examples[label_name][i])
                new_examples["Entailment"].append(entailment_label)

    return new_examples

def process_nli(binary: Dataset, save_dir: str):
    """
    Convert binary dataset to nli dataset.
    """

    logger.info("Processing NLI dataset.")
    nli_dataset = binary.map(convert_to_nli, batched=True, remove_columns=binary["train"].column_names)

    save_path = os.path.join(save_dir, "nli_dataset")
    nli_dataset.save_to_disk(save_path)
    
    return nli_dataset

def process_binary(
    abstract_df: pd.DataFrame,
    binary_label_path: str,
    save_dir: str
) -> Dataset:
    """
    Process dataset for binary labels. Saving to "<save_dir>/binary_dataset"
    """

    logger.info("Processing binary dataset.")
    binary_df = pd.read_csv(binary_label_path)
    binary_data = pd.merge(abstract_df, binary_df, on="PMID")
    binary_data["label"] = binary_data.iloc[:, 4:].astype('float64').values.tolist()

    binary_dataset = {
        "train": Dataset.from_pandas(binary_data[binary_data["Split"] == "Train"]),
        "validation": Dataset.from_pandas(binary_data[binary_data["Split"] =="Validation"]),
        "test": Dataset.from_pandas(binary_data[binary_data["Split"] =="Test"])
    }
    binary_dataset = DatasetDict(binary_dataset)

    save_path = os.path.join(save_dir, "binary_dataset")
    binary_dataset.save_to_disk(save_path)

    return binary_dataset

def process_soft(
    abstract_df: pd.DataFrame,
    soft_label_path: str,
    save_dir: str
) -> Dataset:
    """
    Process dataset for soft labels. Saving to "<save_dir>/soft_dataset"
    """

    logger.info("Processing soft dataset.")
    soft_df = pd.read_csv(soft_label_path)
    soft_data = pd.merge(abstract_df, soft_df, on="PMID")
    soft_data["label"] = soft_data.iloc[:, 4:].astype('float64').values.tolist()

    soft_dataset = {
        "train": Dataset.from_pandas(soft_data[soft_data["Split"] == "Train"]),
        "validation": Dataset.from_pandas(soft_data[soft_data["Split"] =="Validation"]),
        "test": Dataset.from_pandas(soft_data[soft_data["Split"] =="Test"])
    }
    soft_dataset = DatasetDict(soft_dataset)

    save_path = os.path.join(save_dir, "soft_dataset")
    soft_dataset.save_to_disk(save_path)

    return soft_dataset

def process(
    abstract_path: str,
    binary_label_path: str,
    soft_label_path: str,
    save_dir:str
):
    """
    Processes each of binary, soft and nli datasets
    """

    abstract_df = pd.read_csv(abstract_path)
    binary = process_binary(abstract_df, binary_label_path, save_dir)
    _ = process_soft(abstract_df, soft_label_path, save_dir)
    _ = process_nli(binary, save_dir)

if __name__=='__main__':
    args = get_args()
    process(args.abstracts, args.binary_labels, args.soft_labels, args.save_dir)


