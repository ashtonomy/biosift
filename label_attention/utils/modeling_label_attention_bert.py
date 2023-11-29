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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_less_than_1_11
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from .training_args import OptimizerNames, ParallelMode, TrainingArguments

from transformers.models.bert.modeling_bert import *


def BertLabelEmbedding(nn.Module):
    def __init__(self, config, pretrained_model=None, pretrained_tokenizer=None):
        super(LabelEmbedding, self).__init__()

        self.num_labels = config.num_labels
        self.config = config

        self.init_weights(pretrained_model, pretrained_tokenizer)
    
    def init_weights(self, weights = None, model=None, tokenizer=None):
        """
        Initialize label embedding matrix with sentence embeddings of labels.
        Hidden size of config/model should match target model. 
        """
        
        # Tokenize and encode label names
        self.label_names = self.config.label2idx.keys()

        if model is not None and tokenizer is not None:
            tokenized_labels = tokenizer(self.label_names)
        
            # Get hidden representation of label encodings
            model.eval()
            with torch.no_grad():
                init_embeddings = model(tokenized_labels)[1] # CLS TOKEN OUT
        
            self.label_attention_matrix = nn.Parameter(init_embeddings)    # n_labels x hidden_dim

    def forward(self, inputs):
        """
        Calculate cosine distance between each sample and each label
        
        args
            inputs: Tensor of shape (N x hidden_dim)
        returns
            tensor of shape N x num_classes
        """

        # Compute cosine similarity matrix
        eps = 1e-8
        X_n = X.norm(dim=1)[:, None]
        W_n = self.label_attention_matrix.norm(dim=1)[:, None]
        X_norm = X / torch.where(W_n < eps, W_n)
        W_norm = self.label_attention_matrix / torch.where(W_n < eps, eps, W_n)
        cos_sim = torch.mm(X_norm, self.label_attention_matrix.T)

        return cos_sim

class BertForLabelAttendedSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.label_embedding = BertLabelEmbedding(config, self.bert)
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

        label_embedding_output = self.label_embedding.forward(inputs)
        label_embedding_output = self.label_embedding_dropout(label_embedding_output)

        combined_output = pooled_output * label_embedding_output
        logits = self.classifier(combined_output)

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

    def from_pretrained(
        pretrained_model_name_or_path: typing.Union[str, os.PathLike, NoneType],
        *model_args,
        config: typing.Union[transformers.configuration_utils.PretrainedConfig, str, os.PathLike, NoneType] = None,
        cache_dir: typing.Union[str, os.PathLike, NoneType] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: typing.Union[str, bool, NoneType] = None,
        revision: str = 'main',
        use_safetensors: bool = None,
        **kwargs
    ): 
        """
        Overriding from_pretrained for label_attention
        """
        super().from_pretrained(pretrained_model_name_or_path,
                                model_args,
                                config,
                                cache_dir,
                                ignore_mismatched_sizes,
                                force_download,
                                local_files_only,
                                token,
                                revision,
                                use_safetensors,
                                kwargs)

        if os.path.exists(pretrained_model_name_or_path):
            if os.path.exists():
                torch.load(model, "model.pth") 
        else:
            pass # No pretrained weights for embedding layer