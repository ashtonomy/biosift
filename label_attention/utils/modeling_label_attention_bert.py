"""
Bert implementation with Label Attention
"""
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
import typing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import *

import logging

logger = logging.getLogger(__name__)


class BertLabelEmbedding(nn.Module):
    def __init__(self, config, pretrained_model=None, pretrained_tokenizer=None):
        super(BertLabelEmbedding, self).__init__()

        self.num_labels = config.num_labels
        self.config = config
        self.label_names = list(self.config.label2id.keys())
        self.label_attention_matrix = None

        # self.label_attention_matrix = nn.Parameter(torch.zeros((self.num_labels,
        #                                                         self.config.hidden_size)))  # n_labels x hidden_size

        # Set this flag if pretrained weights will be loaded
        self.pretrained_weights = False

        # if not self.pretrained_weights:
        #     logger.info("Initializing label attention weights with model embeddings.")
        #     self.init_weights(pretrained_model, pretrained_tokenizer)
            
    
    def init_weights(self, model, tokenizer):
        """
        Initialize label embedding matrix with sentence embeddings of labels.
        Hidden size of config/model should match target model. 
        """
        
        if model is not None and tokenizer is not None:
            logger.info(f"Initializing to label embeddings:\n{self.label_names}")
            tokenized_labels = tokenizer(self.label_names, padding="max_length", return_tensors="pt")

            # Get hidden representation of label encodings
            model.eval()
            with torch.no_grad():
                init_embeddings = model(**tokenized_labels)[1] # CLS TOKEN OUT
        
            self.label_attention_matrix = nn.Parameter(init_embeddings)    # n_labels x hidden_size

    def forward(self, inputs):
        """
        Calculate cosine distance between each sample and each label
        
        args
            X: Tensor of shape (N x hidden_size)
        returns
            tensor of shape N x num_classes
        """

        # Compute cosine similarity matrix
        eps = 1e-8
        inputs_n = inputs.norm(dim=1)[:, None]
        W_n = self.label_attention_matrix.norm(dim=1)[:, None]
        inputs_norm = inputs / torch.where(inputs_n < eps, eps, inputs_n)
        W_norm = self.label_attention_matrix / torch.where(W_n < eps, eps, W_n)
        cos_sim = torch.mm(inputs_norm, W_norm.T)

        return cos_sim

class BertForLabelAttendedSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        logger.info(f"Initializing model with {self.num_labels} labels")

        self.bert = BertModel(config)
        
        self.label_embedding = BertLabelEmbedding(config)
        self.label_embedding_dropout = nn.Dropout(config.hidden_dropout_prob)        

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        label_embedding_output = self.label_embedding.forward(pooled_output)
        label_embedding_output = self.label_embedding_dropout(label_embedding_output)

        logits = self.classifier(pooled_output)
        logits = logits * label_embedding_output

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """
        Overriding from_pretrained for label_attention weights. 
        """
        model = super().from_pretrained(pretrained_model_name_or_path,
                                        model_args,
                                        config,
                                        cache_dir,
                                        ignore_mismatched_sizes,
                                        force_download = force_download,
                                        local_files_only = local_files_only,
                                        token = token,
                                        revision = revision,
                                        use_safetensors = use_safetensors,
                                        **kwargs)

        weight_path = os.path.join(pretrained_model_name_or_path, "label_embedding.pt")
        if os.path.exists(weight_path):
            logger.info(f"Initializing label embedding weights from {pretrained_model_name_or_path}.")
            model.label_embedding.label_attention_matrix = torch.load(weight_path)
            model.label_embedding.pretrained_weights = True
        elif "tokenizer" in kwargs:
            if "model" in kwargs:
                model.label_embedding.init_weights(kwargs["model"], kwargs["tokenizer"])
            else:
                model.label_embedding.init_weights(model.bert, kwargs["tokenizer"])
        
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Saves base model plus label_embedding in save directory
        """

        super().save_pretrained(save_directory,
                                is_main_process,
                                state_dict,
                                save_function = save_function,
                                push_to_hub = push_to_hub,
                                max_shard_size = max_shard_size,
                                safe_serialization = safe_serialization,
                                variant = variant,
                                token = token,
                                save_peft_format = save_peft_format,
                                **kwargs)


        torch.save(self.label_embedding.state_dict(), os.path.join(save_directory, "label_embedding.pt"))