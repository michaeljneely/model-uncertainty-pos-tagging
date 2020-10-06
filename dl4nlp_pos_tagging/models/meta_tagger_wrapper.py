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
from dl4nlp_pos_tagging.models.meta_wrapper import MetaWrapper
import dl4nlp_pos_tagging.common.utils as utils
import numpy as np
logger = logging.getLogger(__name__)

@Model.register("meta_tagger_wrapper")
class MetaTaggerWrapper(MetaWrapper):

    @overrides
    def forward(
        self,
        tokens,
        metadata,
        tags: torch.LongTensor = None,
        **kwargs
        ) -> Dict[str, torch.Tensor]:

        output_dict = {}
        component_args = {}

        for name in self.component_models:
            component_output = self.component_models[name](
                tokens=tokens,
                metadata=metadata,
                tags=tags
            )
            component_args[name] = component_output.pop("output")
            utils.extend_dictionary_by_namespace(output_dict, name, component_output)

        meta_output = self.meta_model(
            tokens=tokens,
            metadata=metadata,
            tags=tags,
            **component_args,
            **kwargs
        )
        utils.extend_dictionary_by_namespace(output_dict, "meta", meta_output)

        if tags is not None:
            loss = sum(output_dict.pop(f"{k}_loss") for k in self.all_model_keys)
            output_dict["loss"] = loss
            output_dict["actual"] = tags

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a `"tags"` key to the dictionary with the result.
        """
        for name in self.all_model_keys:
            all_predictions = output_dict[f"{name}_class_probabilities"]
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            all_tags = []
            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [
                    self.vocab.get_token_from_index(x, namespace=self.label_namespace)
                    for x in argmax_indices
                ]
                all_tags.append(tags)
            output_dict[f"{name}_tags"] = all_tags
        return output_dict
