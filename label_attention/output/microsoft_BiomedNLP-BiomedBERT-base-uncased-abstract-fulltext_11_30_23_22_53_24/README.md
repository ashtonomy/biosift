---
license: mit
base_model: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
tags:
- generated_from_trainer
model-index:
- name: microsoft_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_11_30_23_22_53_24
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# microsoft_BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_11_30_23_22_53_24

This model is a fine-tuned version of [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1933
- F1 Micro: 0.9545
- F1 Macro: 0.9416
- Precision Micro: 0.9546
- Precision Macro: 0.9441
- Recall Micro: 0.9543
- Recall Macro: 0.9398
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
| 0.325         | 1.0   | 501  | 0.2228          | 0.9487   | 0.9348   | 0.9491          | 0.9428          | 0.9484       | 0.9306       | 0.9424         |
| 0.2544        | 2.0   | 1002 | 0.2009          | 0.9548   | 0.9442   | 0.9478          | 0.9381          | 0.9618       | 0.9506       | 0.9495         |
| 0.2252        | 3.0   | 1503 | 0.1941          | 0.9546   | 0.9425   | 0.9448          | 0.9355          | 0.9646       | 0.9506       | 0.9488         |
| 0.2034        | 4.0   | 2004 | 0.1925          | 0.9549   | 0.9415   | 0.9529          | 0.9430          | 0.9569       | 0.9416       | 0.9485         |
| 0.1872        | 5.0   | 2505 | 0.1933          | 0.9545   | 0.9416   | 0.9546          | 0.9441          | 0.9543       | 0.9398       | 0.9482         |


### Framework versions

- Transformers 4.34.0
- Pytorch 2.1.0+cu121
- Datasets 2.14.5
- Tokenizers 0.14.1
