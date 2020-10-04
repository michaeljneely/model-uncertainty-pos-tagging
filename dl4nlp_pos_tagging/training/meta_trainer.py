from allennlp.training import Trainer, GradientDescentTrainer, BatchCallback, EpochCallback
from allennlp.models import Model
from typing import List, Optional, Dict, Tuple, Any, Union, Iterator, Generator
from allennlp.data import DataLoader
import torch
from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
import math
from allennlp.common.checks import ConfigurationError, check_for_gpu
import re
from allennlp.common import util as common_util
from dl4nlp_pos_tagging.models.meta_wrapper import MetaWrapper
from overrides import overrides
import time
from torch.cuda import amp
import logging
from allennlp.data.dataloader import TensorDict
from allennlp.nn import util as nn_util
from allennlp.common.util import int_to_device
from copy import deepcopy
from dataclasses import dataclass, field
from collections import defaultdict, ChainMap
import allennlp.training.util as training_util
from contextlib import contextmanager
import datetime
import traceback
import os

@dataclass
class ComponentLoss:
    loss: float = field(default=0.0)
    batch_loss: float = field(default=0.0)
    reg_loss: float = field(default=None)
    batch_reg_loss: float = field(default=None)

logger = logging.getLogger(__name__)

class ComponentOptimizer(Registrable):

    default_implementation = "gradient_descent"

    def __init__(
        self,
        name: str,
        model: Model,
        optimizer: Optimizer,
        cuda_device: int,
        grad_norm: Optional[float] = None,
        scaler: Optional[amp.GradScaler] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None
    ) -> "ComponentOptimizer":

        self.name = name
        self.model = model
        self._optimizer = optimizer

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        self._cuda_device = int_to_device(cuda_device)
        self._grad_norm = grad_norm
        self._scaler = scaler
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._loss = {
            'train': ComponentLoss(),
            'validation': ComponentLoss()
        }


    def enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            for parameter in self.model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda grad: nn_util.clamp_tensor(
                            grad, minimum=-self._grad_clipping, maximum=self._grad_clipping
                        )
                    )

    def rescale_gradients(self) -> float:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        Returns the norm of the gradients.
        """
        parameters_to_clip = [p for p in self.model.parameters() if p.grad is not None]

        if self._grad_norm:
            if self._scaler is not None:
                # Need to first unscale gradients in order to clip as usual.
                self._scaler.unscale_(self._optimizer)
            return torch.nn_utils.clip_grad_norm_(parameters_to_clip, self._grad_norm)
        else:
            return torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters_to_clip])
            )

    def get_state(self):
        state = {
            "state": self._optimizer.state_dict()
        }
        if self._learning_rate_scheduler is not None:
            state["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            state["momentum_scheduler"] = self._momentum_scheduler.state_dict()
        return state

    def zero_grad(self):
        self._optimizer.zero_grad()

    def reset_loss(self, key):
        regularization_penalty = self.model.get_regularization_penalty()
        if regularization_penalty is not None:
            self._loss[key] = ComponentLoss(train_reg_loss=0.0, batch_reg_loss=0.0)
        else:
            self._loss[key] = ComponentLoss()

    def _batch_outputs(self, batch: TensorDict, for_training: bool) -> Dict[str, torch.Tensor]:
        """
        Does a forward pass on the given batch and returns the output dictionary that the model
        returns, after adding any specified regularization penalty to the loss (if training).
        """
        batch = nn_util.move_to_device(batch, self._cuda_device)
        output_dict = self.model(**batch)

        if for_training:
            try:
                assert "loss" in output_dict
                regularization_penalty = self.model.get_regularization_penalty()

                if regularization_penalty is not None:
                    output_dict["reg_loss"] = regularization_penalty
                    output_dict["loss"] += regularization_penalty

            except AssertionError:
                if for_training:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " 'loss' key in the output of model.forward(inputs)."
                    )
        return output_dict

    def process_batch_group(
        self,
        batch_group: List[TensorDict],
        for_training: Optional[bool] = True,
        batch_num_total: Optional[int] = None,
        batches_this_epoch: Optional[int] = None,
        retain_graph: Optional[bool] = False
    ) -> Dict[str, torch.Tensor]:

        batch_group_outputs = []
        loss_key = "train" if for_training else "validation"

        for batch in batch_group:
            batch_outputs = self._batch_outputs(batch=batch, for_training=for_training)
            batch_group_outputs.append(batch_outputs)
            loss = batch_outputs.get("loss")
            reg_loss = batch_outputs.get("reg_loss")

            if torch.isnan(loss) and for_training:
                raise ValueError("nan loss encountered during training")

            loss = loss / len(batch_group)
            self._loss[loss_key].batch_loss = loss.item()
            self._loss[loss_key].loss += self._loss[loss_key].batch_loss

            if reg_loss is not None:
                reg_loss = reg_loss / len(batch_group)
                self._loss[loss_key].batch_reg_loss = reg_loss.item()
                self._loss[loss_key].reg_loss += self._loss[loss_key].batch_reg_loss

            if for_training:
                torch.autograd.set_detect_anomaly(True)
                loss.backward(retain_graph=retain_graph)
                # temp.backward(autograd.grad(loss1, temp_d, only_input=False)[0], retain_graph=True)
        batch_grad_norm = self.rescale_gradients()

        # This does nothing if batch_num_total is None or you are using a
        # scheduler which doesn't update per batch.
        if for_training:
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._scaler is not None:
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()

        # Need to prepend the name of the component optimizer to all metrics so they are identifiable
        metrics = {}

        model_metrics = self.model.get_metrics(reset=False)
        for k, v in model_metrics.items():
            metrics[f'{self.name}_{k}'] = v

        if self._loss[loss_key].batch_loss is not None:
            metrics[f"{self.name}_batch_loss"] = self._loss[loss_key].batch_loss

        metrics[f"{self.name}_loss"] = float(self._loss[loss_key].loss / batches_this_epoch) if batches_this_epoch > 0 else 0.0
        if self._loss[loss_key].reg_loss is not None:
            if self._loss[loss_key].batch_reg_loss is not None:
                metrics[f"{self.name}_batch_reg_loss"] = self._loss[loss_key].batch_reg_loss
            metrics[f"{self.name}_reg_loss"] = float(self._loss[loss_key].reg_loss / batches_this_epoch) if batches_this_epoch > 0 else 0.0

        return batch_group_outputs, metrics



    @classmethod
    def from_partial_objects(
        cls,
        name: str,
        model: Model,
        num_epochs: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: float = None,
        grad_clipping: float = None,
        optimizer: Lazy[Optimizer] = None,
        learning_rate_scheduler: Lazy[LearningRateScheduler] = None,
        momentum_scheduler: Lazy[MomentumScheduler] = None,
        tensorboard_writer: Lazy[TensorboardWriter] = None,
    ) -> "ComponentOptimizer":
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer_ = optimizer.construct(model_parameters=parameters)
        if not optimizer_:
            optimizer_ = Optimizer.default(parameters)

        common_util.log_frozen_and_tunable_parameter_names(model)

        learning_rate_scheduler_ = learning_rate_scheduler.construct(
            optimizer=optimizer_, num_epochs=num_epochs, num_steps_per_epoch=batches_per_epoch
        )
        momentum_scheduler_ = momentum_scheduler.construct(optimizer=optimizer_)

        return cls(
            name=name,
            model=model,
            optimizer=optimizer_,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=learning_rate_scheduler_,
            momentum_scheduler=momentum_scheduler_
        )

ComponentOptimizer.register("gradient_descent", constructor="from_partial_objects")(ComponentOptimizer)


@Trainer.register("meta", constructor="from_partial_objects")
class MetaTrainer(Trainer):

    def __init__(
        self,
        model: MetaWrapper,
        component_optimizers: Dict[str, ComponentOptimizer],
        data_loader: DataLoader,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        tensorboard_writer: TensorboardWriter = None,
        moving_average: Optional[MovingAverage] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
    ):
        super().__init__(serialization_dir, cuda_device)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.data_loader = data_loader
        self._validation_data_loader = validation_data_loader
        self.component_optimizers = component_optimizers

        if patience is None:  # no early stopping
            if validation_data_loader is not None:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(serialization_dir)

        self._batch_callbacks = batch_callbacks or []
        self._epoch_callbacks = epoch_callbacks or []
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # `_enable_activation_logging`.
        self._batch_num_total = 0

        self._tensorboard = tensorboard_writer or TensorboardWriter(serialization_dir)
        self._tensorboard.get_batch_num_total = lambda: self._batch_num_total
        self._tensorboard.enable_activation_logging(self.model)

        self._last_log = 0.0  # time of last logging

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        # Enable automatic mixed precision training.
        self._scaler: Optional[amp.GradScaler] = None
        self._use_amp = use_amp
        if self._use_amp:
            if self.cuda_device == torch.device("cpu"):
                raise ValueError("Using AMP requires a cuda device")
            self._scaler = amp.GradScaler()

        self._is_master = True
        self._pytorch_model = self.model


    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        for optimizer in self.component_optimizers.values():
            optimizer.enable_gradient_clipping()

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for callback in self._epoch_callbacks:
            callback(self, metrics={}, epoch=-1, is_master=self._master)

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            for key, value in train_metrics.items():
                if key.startswith("gpu_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)
                elif key.startswith("worker_") and key.endswith("_memory_MB"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_metrics = self._validation_loss(epoch)
                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[f"meta_{self._validation_metric}"]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            for name, sub_model in self._pytorch_model.component_models.items():
                component_optimizer = self.component_optimizers[name]
                metric = val_metrics[f"{name}_{self._validation_metric}"]
                if component_optimizer._learning_rate_scheduler:
                    component_optimizer._learning_rate_scheduler.step(metric)
                if component_optimizer._momentum_scheduler:
                    component_optimizer._momentum_scheduler.step(metric)

            if self._master:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            for callback in self._epoch_callbacks:
                callback(self, metrics=metrics, epoch=epoch, is_master=self._master)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info(f"Epoch: {epoch}/{self._num_epochs - 1}")
        cpu_memory_usage = []
        for worker, memory in common_util.peak_memory_mb().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage MB: {memory}")
        gpu_memory_usage = []
        for gpu, memory in common_util.gpu_memory_mb().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        for component_optimizer in self.component_optimizers.values():
            component_optimizer.reset_loss('train')

        self.model.train()

        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        batch_group_generator_tqdm = Tqdm.tqdm(
            batch_group_generator, total=num_training_batches
        )

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False

        for batch_group in batch_group_generator_tqdm:

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            for component_optimizer in self.component_optimizers.values():
                component_optimizer.zero_grad()

            batch_group_metrics = []

            meta_batch = deepcopy(batch_group)

            # Train the Sub Models first
            for name, sub_model in self._pytorch_model.component_models.items():
                component_optimizer = self.component_optimizers[name]
                batch_group_outputs, metrics = component_optimizer.process_batch_group(
                    batch_group,
                    True,
                    batch_num_total,
                    batches_this_epoch,
                    True
                )
                batch_group_metrics.append(metrics)

                for i, batch_outputs in enumerate(batch_group_outputs):
                    component_output = batch_outputs["output"]
                    component_output = component_output.detach()
                    meta_batch[i][name] = component_output

            meta_optimizer = self.component_optimizers["meta"]
            meta_batch_outputs, meta_metrics = meta_optimizer.process_batch_group(
                meta_batch,
                True,
                batch_num_total,
                batches_this_epoch,
                False
            )

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            batch_group_metrics.append(meta_metrics)

            all_metrics = ChainMap(*batch_group_metrics)

            description = training_util.description_from_metrics(all_metrics)
            batch_group_generator_tqdm.set_description(description, refresh=False)

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory

        return all_metrics

    def _validation_loss(self, epoch: int) -> Tuple[float, float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_data_loader is not None:
            validation_data_loader = self._validation_data_loader
        else:
            raise ConfigurationError(
                "Validation results cannot be calculated without a validation_data_loader"
            )

        val_generator_tqdm = Tqdm.tqdm(validation_data_loader)

        for component_optimizer in self.component_optimizers.values():
            component_optimizer.reset_loss('validation')

        batches_this_epoch = 0
        done_early = False

        for batch in val_generator_tqdm:
            batches_this_epoch += 1

            batch_metrics = []
            batch_group = [batch]
            meta_batch = deepcopy(batch_group)

            # Train the Sub Models first
            for name, sub_model in self._pytorch_model.component_models.items():
                component_optimizer = self.component_optimizers[name]
                batch_group_outputs, metrics = component_optimizer.process_batch_group(
                    batch_group,
                    for_training=False,
                    batches_this_epoch=batches_this_epoch
                )
                batch_metrics.append(metrics)

                for i, batch_outputs in enumerate(batch_group_outputs):
                    meta_batch[i][name] = batch_outputs["output"]

            meta_optimizer = self.component_optimizers["meta"]
            meta_batch_outputs, meta_metrics = meta_optimizer.process_batch_group(
                meta_batch,
                for_training=False,
                batches_this_epoch=batches_this_epoch
            )
            batch_metrics.append(meta_metrics)

            all_metrics = ChainMap(*batch_metrics)
            description = training_util.description_from_metrics(all_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return all_metrics

    @contextmanager
    def get_checkpoint_state(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        if self._moving_average is not None:
            # Assigning average value to model parameters.  The checkpointer will call
            # `restore_state_after_checkpointing` when it is done to put this back to what it was.
            self._moving_average.assign_average_value()

        model_state = self.model.state_dict()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizers": {k: v.get_state() for k, v in self.component_optimizers.items()},
            "batch_num_total": self._batch_num_total,
        }
        try:
            yield model_state, training_states
        finally:
            if self._moving_average is not None:
                self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`
        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.
        # Returns
        epoch: `int`
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)

        for name, cm in self.component_optimizers.items():
            state = training_state["optimizers"][name]
            cm._optimizer.load_state_dict(state["state"])

            if (
                cm._learning_rate_scheduler is not None
                and "learning_rate_scheduler" in state
            ):
                cm._learning_rate_scheduler.load_state_dict(state["learning_rate_scheduler"])

            if cm._momentum_scheduler is not None and "momentum_scheduler" in state:
                cm._momentum_scheduler.load_state_dict(state["momentum_scheduler"])

            training_util.move_optimizer_to_cuda(cm._optimizer)

        # Currently the `training_state` contains a serialized `MetricTracker`.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked `val_metric_per_epoch`.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return


    @classmethod
    def from_partial_objects(
        cls,
        model: MetaWrapper,
        serialization_dir: str,
        data_loader: DataLoader,
        validation_data_loader: DataLoader = None,
        patience: int = None,
        validation_metric: str = "-loss",
        num_epochs: int = 20,
        cuda_device: Optional[Union[int, torch.device]] = None,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        no_grad: List[str] = None,
        component_optimizers: Dict[str, Lazy[ComponentOptimizer]] = None,
        tensorboard_writer: Lazy[TensorboardWriter] = None,
        moving_average: Lazy[MovingAverage] = None,
        checkpointer: Lazy[Checkpointer] = None,
        batch_callbacks: List[BatchCallback] = None,
        epoch_callbacks: List[EpochCallback] = None,
    ) -> "Trainer":

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)
            model.meta_model = model.meta_model.cuda(cuda_device)
            for name in model.component_models:
                model.component_models[name] = model.component_models[name].cuda(cuda_device)

        if no_grad:
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in no_grad):
                    parameter.requires_grad_(False)


        batches_per_epoch: Optional[int]
        try:
            batches_per_epoch = len(data_loader)
            batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)
        except TypeError:
            batches_per_epoch = None

        sub_models = model.get_all_models()

        for name, sub_model in sub_models.items():
            component_optimizers[name] = component_optimizers[name].construct(
                name=name,
                model=sub_model,
                num_epochs=num_epochs,
                batches_per_epoch=batches_per_epoch,
                cuda_device=cuda_device
            )

        all_parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        moving_average_ = moving_average.construct(parameters=all_parameters)

        checkpointer_ = checkpointer.construct() or Checkpointer(serialization_dir)
        tensorboard_writer_ = tensorboard_writer.construct() or TensorboardWriter(serialization_dir)

        return cls(
            model=model,
            component_optimizers=component_optimizers,
            data_loader=data_loader,
            patience=patience,
            validation_metric=validation_metric,
            validation_data_loader=validation_data_loader,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            checkpointer=checkpointer_,
            moving_average=moving_average_,
            cuda_device=cuda_device,
            tensorboard_writer=tensorboard_writer_,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
            use_amp=use_amp
        )

