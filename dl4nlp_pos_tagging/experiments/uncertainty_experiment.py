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
        self.dataset = self._batch_dataset(instances)
        self.num_batches = math.ceil(len(instances) / self.batch_size)

        # Dataframes
        self.results = None


    def _batch_dataset(
        self,
        unlabeled_instances: List[Instance]
    ) -> Generator[InstanceBatch, None, None]:
        ids = iter(range(len(unlabeled_instances)))
        for b_id, instance_batch in enumerate(utils.batch(unlabeled_instances, self.batch_size)):

            batch_outputs = self.predictor._model.forward_on_instances(instance_batch)

            actual_labels = [instance["actual"] for instance in batch_outputs]

            batch_ids = [next(ids) for _ in range(len(instance_batch))]

            labeled_batch = [ \
                self.predictor.predictions_to_labeled_instances(instance, outputs)[0] \
                for instance, outputs in zip(instance_batch, batch_outputs) \
            ]

            yield (batch_ids, labeled_batch, actual_labels)

    def _calculate_uncertainty_batch(self, batch: InstanceBatch, progress_bar: Tqdm = None) -> None:
        # TODO
        return

    def calculate_uncertainty(self, force: bool = False) -> None:
        pkl_exists = os.path.isfile(os.path.join(self.serialization_dir, 'uncertainty.pkl'))

        if pkl_exists and not force:
            self.logger.info("Uncertainty data exists and force was not specified. Loading from disk...")
            self.results = pd.read_pickle(os.path.join(self.serialization_dir, 'uncertainty.pkl'))
        else:
            uncertainty_df = defaultdict(list)
            self.logger.info('Calculating uncertainty...')

            progress_bar = Tqdm.tqdm(total=self.num_batches)

            for batch in self.dataset:
                # TODO
                pass

            self.results = pd.DataFrame(uncertainty_df)
            utils.write_frame(self.results, self.serialization_dir, 'uncertainty')

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

        return cls(
            serialization_dir=serialization_dir,
            predictor=predictor,
            instances=test_instances,
            batch_size=batch_size,
            logger=logger
        )

UncertaintyExperiment.register("default", constructor="from_partial_objects")(UncertaintyExperiment)
