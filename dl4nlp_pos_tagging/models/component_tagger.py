from allennlp.models import Model, SimpleTagger
from allennlp.modules import FeedForward, Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder
from allennlp.data import Vocabulary, TextFieldTensors
from overrides import overrides
import torch
import allennlp.nn.util as nn_util
from allennlp.common import JsonDict
from typing import Optional, List, Dict, Any, Tuple
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import InitializerApplicator
import torch.nn.functional as F

# A wrapper around the SimpleTagger that returns the encoded output in the output_dict

@Model.register("component_tagger")
class ComponentTagger(SimpleTagger):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ):
        super().__init__(
            vocab=vocab,
            text_field_embedder=text_field_embedder,
            encoder=encoder,
            calculate_span_f1=calculate_span_f1,
            label_encoding=label_encoding,
            label_namespace=label_namespace,
            verbose_metrics=verbose_metrics,
            initializer=initializer,
            **kwargs
        )

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = nn_util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {
            "output": encoded_text,
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
