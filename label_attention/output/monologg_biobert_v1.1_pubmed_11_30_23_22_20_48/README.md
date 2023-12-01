---
base_model: monologg/biobert_v1.1_pubmed
tags:
- generated_from_trainer
model-index:
- name: monologg_biobert_v1.1_pubmed_11_30_23_22_20_48
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# monologg_biobert_v1.1_pubmed_11_30_23_22_20_48

This model is a fine-tuned version of [monologg/biobert_v1.1_pubmed](https://huggingface.co/monologg/biobert_v1.1_pubmed) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1951
- F1 Micro: 0.9542
- F1 Macro: 0.9420
- Precision Micro: 0.9527
- Precision Macro: 0.9424
- Recall Micro: 0.9557
- Recall Macro: 0.9421
- Combined Score: 0.9482

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | F1 Micro | F1 Macro | Precision Micro | Precision Macro | Recall Micro | Recall Macro | Combined Score |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:--------:|:---------------:|:---------------:|:------------:|:------------:|:--------------:|
| 0.3254        | 1.0   | 501  | 0.2267          | 0.9491   | 0.9368   | 0.9429          | 0.9347          | 0.9554       | 0.9405       | 0.9432         |
| 0.2548        | 2.0   | 1002 | 0.2047          | 0.9521   | 0.9403   | 0.9431          | 0.9332          | 0.9613       | 0.9482       | 0.9464         |
| 0.2255        | 3.0   | 1503 | 0.1953          | 0.9546   | 0.9430   | 0.9441          | 0.9351          | 0.9652       | 0.9518       | 0.9490         |
| 0.2035        | 4.0   | 2004 | 0.1931          | 0.9536   | 0.9409   | 0.9543          | 0.9445          | 0.9530       | 0.9383       | 0.9474         |
| 0.1855        | 5.0   | 2505 | 0.1951          | 0.9542   | 0.9420   | 0.9527          | 0.9424          | 0.9557       | 0.9421       | 0.9482         |


### Framework versions

- Transformers 4.34.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.5
- Tokenizers 0.14.1
