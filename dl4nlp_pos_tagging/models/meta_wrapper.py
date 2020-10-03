import logging
import os
from os import PathLike
import re
from typing import Dict, List, Set, Type, Optional, Union

import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator
from overrides import overrides
import itertools

logger = logging.getLogger(__name__)

@Model.register("meta_wrapper")
class MetaWrapper(Model):
    """
    A MetaWrapper is a Model which consists of component models and a meta model. The encoded output of each component
    model is passed to the meta model for a final prediction.

    Every component model must return an `output` key in its output dictionary
    """

    def __init__(
        self,
        vocab: Vocabulary,
        component_models: Dict[str, Model],
        meta_model: Model
    ):
        super().__init__(vocab)
        self.component_models = component_models

        if "meta" in component_models.keys():
            raise ConfigurationError("Reserved name 'meta' cannot be used for a component model.")

        self.meta_model = meta_model

    def get_all_models(self):
        meta_model = {"meta": self.meta_model}
        return {**meta_model, **self.component_models}

    @overrides
    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        all_parameters = [m.named_parameters() for m in [self.meta_model] + list(self.component_models.values())]
        return list(itertools.chain.from_iterable(all_parameters))

    @overrides
    def forward(
        self,
        **kwargs
        ) -> Dict[str, torch.Tensor]:
        # This should only be called if you are using a single optimizer
        raise NotImplementedError('To be implemented for joint optimization scheme with a single optimizer')
