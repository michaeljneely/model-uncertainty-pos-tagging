from overrides import overrides
from typing import Optional

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.predictors import Predictor
import torch

@Predictor.register("mc_dropout_sentence_tagger")
class MCDropoutSentenceTaggerPredictor(SentenceTaggerPredictor):


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
