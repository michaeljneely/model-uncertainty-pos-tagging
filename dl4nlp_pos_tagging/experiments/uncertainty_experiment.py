from collections import defaultdict
import itertools
import logging
import math
import os
import random
from os import PathLike
from typing import Any, Dict, List, Generator, Optional, Tuple, Union

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.interpret import SaliencyInterpreter
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import numpy as np
import pandas as pd
import statistics
import torch
from allennlp.common.checks import ConfigurationError, check_for_gpu
from dl4nlp_pos_tagging.config import Config
import dl4nlp_pos_tagging.common.utils as utils
InstanceBatch = Tuple[List[int], List[Instance], List[LabelField]]
import dl4nlp_pos_tagging.common.plotting as plotting

# TODO: Extend skeleton infrastructure

class UncertaintyExperiment(Registrable):

    default_implementation = "default"

    def __init__(
        self,
        serialization_dir: PathLike,
        predictor: Predictor,
        instances: List[Instance],
        batch_size: int,
        logger: Optional[logging.Logger] = None
    ):
        self.serialization_dir = serialization_dir
        self.predictor = predictor
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(Config.logger_name)

        # Dataset
        self.num_instances = len(instances)
        self.dataset = self._batch_dataset(instances)
        self.num_batches = math.ceil(len(instances) / self.batch_size)

        # Dataframes
        self.results = None
        plotting.set_styles()

    def _batch_dataset(
        self,
        unlabeled_instances: List[Instance]
    ) -> Generator[InstanceBatch, None, None]:
        ids = iter(range(len(unlabeled_instances)))
        for b_id, instance_batch in enumerate(utils.batch(unlabeled_instances, self.batch_size)):

            batch_outputs = self.predictor._model.forward_on_instances(instance_batch)

            actual_labels = [instance["actual"] for instance in batch_outputs]

            batch_ids = [next(ids) for _ in range(len(instance_batch))]

            predictions = [self.predictor.predict_instance(instance) for instance in instance_batch]

            yield (batch_ids, predictions, actual_labels)

    def _calculate_uncertainty_batch(self, batch: InstanceBatch, progress_bar: Tqdm = None) -> None:
        uncertainty_df = defaultdict(list)
        ids, predictions, labels = batch
        for idx, prediction, label in zip(ids, predictions, labels):
            for w, word in enumerate(prediction['words']):
                for model in self.predictor._model.all_model_keys:

                    tag_mean_probability = prediction[f'{model}_class_probabilities'][w]
                    tag_std_probability  = prediction[f'{model}_class_prob_std'][w]
                    actual_label_idx = label[w]
                    predicted_label_idx = np.argmax(tag_mean_probability)

                    uncertainty_df['instance_id'].append(idx)
                    uncertainty_df['model'].append(model)
                    uncertainty_df['word'].append(word)

                    uncertainty_df['actual_tag'].append(
                        self.predictor._model.vocab.get_token_from_index(
                            actual_label_idx,
                            namespace=self.predictor._model.label_namespace
                        )
                    )
                    uncertainty_df['actual_confidence_mean'].append(tag_mean_probability[actual_label_idx])
                    uncertainty_df['actual_confidence_std'].append(tag_std_probability[actual_label_idx])


                    uncertainty_df['predicted_tag'].append(
                        self.predictor._model.vocab.get_token_from_index(
                            predicted_label_idx,
                            namespace=self.predictor._model.label_namespace
                        )
                    )

                    uncertainty_df['predicted_confidence_mean'].append(tag_mean_probability[predicted_label_idx])
                    uncertainty_df['predicted_confidence_std'].append(tag_std_probability[predicted_label_idx])
            progress_bar.update(1)
        return uncertainty_df

    def calculate_uncertainty(self, force: bool = False) -> None:
        pkl_exists = os.path.isfile(os.path.join(self.serialization_dir, 'uncertainty.pkl'))

        if pkl_exists and not force:
            self.logger.info("Uncertainty data exists and force was not specified. Loading from disk...")
            self.results = pd.read_pickle(os.path.join(self.serialization_dir, 'uncertainty.pkl'))
        else:
            uncertainty_df = defaultdict(list)
            self.logger.info('Calculating uncertainty...')

            progress_bar = Tqdm.tqdm(total=self.num_instances)

            for batch in self.dataset:
                uncertainty_scores = self._calculate_uncertainty_batch(batch, progress_bar)
                for k, v in uncertainty_scores.items():
                    uncertainty_df[k].extend(v)

            self.results = pd.DataFrame(uncertainty_df)
            utils.write_frame(self.results, self.serialization_dir, 'uncertainty')

    def _plot_confusion_matrix_by_model(self):
        incorrect = self.results.copy()
        incorrect = incorrect[['predicted_tag', 'actual_tag', 'model']]
        incorrect['correct'] = (incorrect['predicted_tag'] != incorrect['actual_tag']).astype(int)
        for model in self.predictor._model.all_model_keys:
            model_confusion_matrix = incorrect[incorrect['model'] == model]
            model_confusion_matrix = model_confusion_matrix.pivot_table(
                index="predicted_tag",
                columns="actual_tag",
                values="correct",
                aggfunc=np.sum
            )
            fig, ax = plotting.new_figure()
            plotting.heatmap(
                frame=model_confusion_matrix,
                ax=ax,
                annotate=True
            )
            plotting.annotate(
                fig=fig,
                ax=ax,
                xlabel="Actual Tag",
                ylabel="Predicted Tag"
            )
            plotting.save_figure(self.serialization_dir, f'confusion_matrix_{model}')

    def _plot_confidence_by_tag(self):
        fig, ax = plotting.new_figure()
        plotting.grouped_boxplot(
            x='predicted_tag',
            y='predicted_prob_mean',
            hue='model',
            frame=self.results
        )
        plotting.annotate(
            fig=fig,
            ax=ax,
            xlabel='POS tag',
            ylabel='Average Confidence',
            title='Model Confidence'
        )
        plotting.save_figure(self.serialization_dir, 'confidence_by_model')

    def generate_artifacts(self):
        self._plot_confidence_by_tag()
        self._plot_confusion_matrix_by_model()

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: PathLike,
        test_data_path: PathLike,
        predictor_type: str,
        batch_size: int,
        cuda_device: Optional[Union[int, torch.device]] = None,
        nr_instances: Optional[int] = 0
    ):
        logger = logging.getLogger(Config.logger_name)

        if cuda_device is None:
            from torch import cuda

            if cuda.device_count() > 0:
                cuda_device = 0
            else:
                cuda_device = -1

        check_for_gpu(cuda_device)


        archive = load_archive(os.path.join(serialization_dir, 'model.tar.gz'), cuda_device=cuda_device)

        predictor = Predictor.from_archive(archive, predictor_type)

        test_instances = list(predictor._dataset_reader.read(test_data_path))
        if nr_instances:
            logger.info(f'Selecting a random subset of {nr_instances} for interpretation')
            test_instances = random.sample(test_instances, min(len(test_instances), nr_instances))

        return cls(
            serialization_dir=serialization_dir,
            predictor=predictor,
            instances=test_instances,
            batch_size=batch_size,
            logger=logger
        )

UncertaintyExperiment.register("default", constructor="from_partial_objects")(UncertaintyExperiment)
