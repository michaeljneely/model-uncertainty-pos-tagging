from overrides import overrides
from typing import Optional

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import SequenceLabelField, TextField, FlagField
from allennlp.models import Model
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors import Predictor
import torch
import numpy as np
from typing import Any, Dict, List, Tuple
from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from collections import defaultdict

@Predictor.register("mc_dropout_sentence_tagger")
class MCDropoutSentenceTaggerPredictor(Predictor):


    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        language: Optional[str] = "en_core_web_sm",
        nr_samples: Optional[int] = 250
    ):
        super().__init__(model, dataset_reader, language)
        self.nr_samples = nr_samples


    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:

        total = torch.zeros((self.nr_samples), device=self._model.device)
        for s in range(self.nr_samples):
            pass


    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, np.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        text_field: TextField = instance["tokens"]
        for name in self._model.all_model_keys:
            predicted_tags = np.argmax(outputs[f"{name}_class_probabilities"], axis=-1)[:len(text_field)].tolist()
            new_instance.add_field(f"{name}_tags", SequenceLabelField(predicted_tags, text_field), self._model.vocab)

        return [new_instance]
