from typing import Dict, Iterable, List, Optional, Any, Tuple, Union
from overrides import overrides

import numpy as np
import torch

from allennlp.common import JsonDict
from allennlp.data import Batch, Instance
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model, SimpleTagger
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder
import allennlp.nn.util as nn_util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure, SpanBasedF1Measure
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import InitializerApplicator
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from allennlp.common.checks import check_dimensions_match, ConfigurationError

@Model.register("meta_tagger")
class MetaTagger(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ):
        super().__init__(vocab, **kwargs)

        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.tag_projection_layer = TimeDistributed(
            Linear(self.encoder.get_output_dim(), self.num_classes)
        )

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }

        # We keep calculate_span_f1 as a constructor argument for API consistency with
        # the CrfTagger, even it is redundant in this class
        # (label_encoding serves the same purpose).
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )
        else:
            self._f1_metric = None

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        character: torch.Tensor,
        word: torch.Tensor,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:

        batch_size, sequence_length, _ = character.size()
        mask = nn_util.get_text_field_mask(tokens)

        cw_encoding = torch.cat((character, word), dim=-1)
        combined_context_sensitive_encoding = self.encoder(cw_encoding, mask)

        logits = self.tag_projection_layer(combined_context_sensitive_encoding)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {
            "logits": logits,
            "class_probabilities": class_probabilities
        }

        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                tag_mask = mask & (tags != o_tag_index)
            else:
                tag_mask = mask
            loss = nn_util.sequence_cross_entropy_with_logits(logits, tags, tag_mask)
            for metric in self.metrics.values():
                metric(logits, tags, mask)
            if self.calculate_span_f1:
                self._f1_metric(logits, tags, mask)
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict
