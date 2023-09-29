from transformers import Trainer
import torch
from torch.nn import functional as F
import logging
import math
import os
import warnings
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Union
from weakref import proxy

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.utilities import _log_hyperparams
from pytorch_lightning.loops import _PredictionLoop, _TrainingEpochLoop
from pytorch_lightning.loops.evaluation_loop import _EvaluationLoop
from pytorch_lightning.loops.fit_loop import _FitLoop
from pytorch_lightning.loops.utilities import _parse_loop_limits, _reset_progress
from pytorch_lightning.plugins import PLUGIN_INPUT, PrecisionPlugin
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from pytorch_lightning.trainer import call, setup
from pytorch_lightning.trainer.configuration_validator import _verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
    _LITERAL_WARN,
    _PRECISION_INPUT,
    _PRECISION_INPUT_STR,
)
from pytorch_lightning.trainer.connectors.callback_connector import _CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import _DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import _LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.utilities import GradClipAlgorithmType, parsing
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerConfig,
    TRAIN_DATALOADERS,
)


class DistilationTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.float_labels = None

    def fit(
        self,
        model: "pl.LightningModule",
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        self.float_labels = model.train_dataloader()
        super().fit(model, val_dataloaders, datamodule, ckpt_path)

    # def compute_loss(self, model, inputs, T, alpha):
    #     # implement custom logic here
    #
    #     # calculate the T5 model probabilities over the input
    #     T5_probabilities = model.calculate_T5_probabilities(inputs)
    #     logits = T5_probabilities.logits
    #
    #     # Calculate KL loss between the T5 probabilities and the desired probabilities.
    #     # custom_loss = torch.nn.KLDivLoss()(F.softmax(logits / T, dim=1),
    #     #                             F.softmax(labels_expanded / T, dim=1)) * (alpha * T * T) + \
    #     #        F.nll_loss(logits, labels) * (1. - alpha)
    #
    #     custom_loss = torch.nn.KLDivLoss()(F.log_softmax(T5_probabilities / T, dim=1),
    #                                 F.softmax(self.float_labels / T, dim=1)) * (alpha * T * T) + \
    #            F.nll_loss(logits, self.float_labels) * (1. - alpha)
    #     # custom_loss = ...
    #     return custom_loss
