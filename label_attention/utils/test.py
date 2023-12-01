import torch
from modeling_label_attention_bert import BertForLabelAttendedSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import os

if __name__ == '__main__':
    model = BertForLabelAttendedSequenceClassification.from_pretrained("/scratch/taw2/test_model")
    tokenizer = AutoTokenizer.from_pretrained("AshtonIsNotHere/biobert_v1.1_biosift")
    raw_ds = load_dataset("AshtonIsNotHere/biosift")

    def proc_func(examples):
        result = tokenizer(examples["Abstract"], padding="max_length", truncation=True, return_tensors='pt')
        return result

    proc_ds = raw_ds["test"].map(proc_func, batched=True, remove_columns = raw_ds["test"].column_names)

    output = model(torch.tensor(proc_ds["input_ids"]), torch.tensor(proc_ds["attention_mask"]), torch.tensor(proc_ds["token_type_ids"]))
    print(output)
