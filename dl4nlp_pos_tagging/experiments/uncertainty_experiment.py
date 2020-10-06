from collections import defaultdict
from copy import copy
import itertools
from functools import partial
import logging
import math
import os
import random
from os import PathLike
import statistics
from typing import Any, Dict, List, Generator, Optional, Tuple, Union

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.interpret import SaliencyInterpreter
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats
import torch

from dl4nlp_pos_tagging.config import Config
import dl4nlp_pos_tagging.common.utils as utils
import dl4nlp_pos_tagging.common.plotting as plotting

InstanceBatch = Tuple[List[int], List[Instance], List[LabelField]]


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
        # plotting.set_styles()

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
                    uncertainty_df['word_id'].append(w)
                    uncertainty_df['model'].append(model)
                    uncertainty_df['word'].append(word)

                    uncertainty_df['actual_tag'].append(
                        self.predictor._model.vocab.get_token_from_index(
                            actual_label_idx,
                            namespace=self.predictor._model.label_namespace
                        )
                    )

                    uncertainty_df['predicted_tag'].append(
                        self.predictor._model.vocab.get_token_from_index(
                            predicted_label_idx,
                            namespace=self.predictor._model.label_namespace
                        )
                    )

                    uncertainty_df['actual_confidence_mean'].append(tag_mean_probability[actual_label_idx])
                    uncertainty_df['actual_confidence_std'].append(tag_std_probability[actual_label_idx])
                    uncertainty_df['predicted_confidence_mean'].append(tag_mean_probability[predicted_label_idx])
                    uncertainty_df['predicted_confidence_std'].append(tag_std_probability[predicted_label_idx])

                    uncertainty_df['mean_probability_distribution'].append(tag_mean_probability)

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

    def _plot_uncertainty_by_disagreement(self):
        df = self.results.copy()
        char_mask = df['model'].values == 'character'
        word_mask = df['model'].values == 'word'
        meta_mask = df['model'].values == 'meta'
        char_df = df[char_mask]
        word_df = df[word_mask]
        meta_df = df[meta_mask]
        meta_df['dist'] = list(map(partial(distance.jensenshannon, base=2.0), char_df['mean_probability_distribution'], word_df['mean_probability_distribution']))
        corr = stats.pearsonr(meta_df['dist'], meta_df['predicted_confidence_std'])
        print(f'Corelation between JS Divergence of char/word models and the meta model\'s uncertainty is: {corr}')
        fig, ax = plotting.new_figure()
        plotting.displot(
            meta_df,
            x="dist",
            y="predicted_confidence_std",
            kind="kde",
            fill=True,
        )
        plt.xlabel("JS Divergence of Character and Word Model Predictive Distributions")
        plt.ylabel("Meta Model Uncertainty")
        plotting.save_figure(self.serialization_dir, f'uncertainty_by_disagreement_b')

    def _plot_confidence_by_tag(self):
        fig, ax = plotting.new_figure()
        plotting.grouped_boxplot(
            x='predicted_tag',
            y='predicted_confidence_mean',
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

    def _latex_table_confidence_by_tags(self):
        table_string = [
            r"\small{\begin{table}[h!]",
            r"\centering",
            r"\begin{tabular}{|l|llc|}",
            r"\hline"
            r"&                       &      \textbf{Model}                 &  \\ \cline{2-4} ",
            r"\textbf{Tag}& \multicolumn{1}{l|}{\textbf{Char}} & \multicolumn{1}{l|}{\textbf{Word}} &  \textbf{Meta}\\ \hline\hline"
        ]
        confidence_by_tag = {k: {} for k in self.predictor._model.vocab._token_to_index[self.predictor._model.label_namespace].keys()}
        confidence_frame = self.results.groupby(['model', 'predicted_tag'], as_index=False).aggregate({
            'predicted_confidence_std': ['mean', 'std']
        })
        confidence_frame.columns = confidence_frame.columns.droplevel()

        confidence_frame.columns = ['model', 'predicted_tag', 'mean_uncertainty', 'std_uncertainty']

        for row in confidence_frame.itertuples(index=False):
            model, predicted_tag, mu, sigma = row
            confidence_by_tag[predicted_tag][model] = (mu, sigma)

        def _format_tuple(mu, sigma):
            mu, sigma = round(mu, 4), round(sigma, 4)
            mu, sigma = utils.strip_preceding_decimal_zero(mu), utils.strip_preceding_decimal_zero(sigma)
            return mu, sigma

        line_string = (
            r"!!TAG!!  & \multicolumn{1}{c|}{!!MU_CHARACTER!!} "\
            r"& \multicolumn{1}{c|}{!!MU_WORD!!} "\
            r"& !!MU_META!!\\ \hline"\
        )
        for tag, confidence_dict in confidence_by_tag.items():
            if not tag[0].isalpha():
                continue
            tag = tag.replace("$", "\$")
            new_line = copy(line_string)
            zipped = []
            for model in self.predictor._model.all_model_keys:
                mu, sigma = confidence_dict.get(model, ("N/A", "N/A"))
                if mu != "N/A":
                    mu, sigma = _format_tuple(mu, sigma)
                zipped.append((mu, f"!!MU_{model.upper()}!!", sigma, f"!!SIGMA_{model.upper()}!!"))
            for (mu, mu_label, sigma, sigma_label) in zipped:
                 new_line = new_line.replace(mu_label, mu)
                 new_line = new_line.replace(sigma_label, sigma)
            new_line = new_line.replace('!!TAG!!', tag)
            table_string.append(new_line)
        table_string.extend([
            r"\end{tabular}",
            r"\caption{Mean uncertainty of the character, word and meta model per part-of-speech tag.}",
            r"\label{tab:uncertainty-per-tag}",
            r"\end{table}}",
        ])
        with open(os.path.join(self.serialization_dir, 'tag_confidence_by_model.tex'), 'w+') as handle:
            for line in table_string:
                handle.write(line + '\n')

    def _announce_accuracy(self):
        accuracy = self.results.copy()
        accuracy['accuracy'] = accuracy['predicted_tag'] == accuracy['actual_tag']
        accuracy = accuracy[['model', 'accuracy']].groupby('model').mean()
        print(accuracy)

    def generate_artifacts(self):
        self._plot_confidence_by_tag()
        self._latex_table_confidence_by_tags()
        self._report_meta_disagreement()
        self._announce_accuracy()
        self._plot_uncertainty_by_disagreement()

    @classmethod
    def from_partial_objects(
        cls,
        serialization_dir: PathLike,
        test_data_path: PathLike,
        predictor_type: str,
        batch_size: int,
        cuda_device: Optional[Union[int, torch.device]] = None,
        nr_instances: Optional[int] = None,
        nr_inference_samples: Optional[int] = 250
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
        predictor.nr_samples = nr_inference_samples
        predictor.batch_size = batch_size

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
