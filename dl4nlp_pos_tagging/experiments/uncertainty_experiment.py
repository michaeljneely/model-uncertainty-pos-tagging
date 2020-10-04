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
        nr_instances: int,
        logger: Optional[logging.Logger] = None
    ):
        self.serialization_dir = serialization_dir
        self.predictor = predictor
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(Config.logger_name)

        # Dataframe
        self.results = None


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
            nr_instances=nr_instances,
            logger=logger
        )

UncertaintyExperiment.register("default", constructor="from_partial_objects")(UncertaintyExperiment)
