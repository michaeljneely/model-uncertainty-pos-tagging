from overrides import overrides
from typing import Optional

from allennlp.common.util import JsonDict, sanitize
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
from allennlp.modules import InputVariationalDropout

@Predictor.register("mc_dropout_sentence_tagger")
class MCDropoutSentenceTaggerPredictor(Predictor):


    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        language: Optional[str] = "en_core_web_sm",
        nr_samples: Optional[int] = 250,
        batch_size: Optional[int] = 32
    ):
        super().__init__(model, dataset_reader, language)
        self.nr_samples = nr_samples
        self.batch_size = batch_size
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"words"` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:

        sub_models = self._model.get_all_models()

        # turn on dropout
        for model in sub_models.values():
            for module in model.modules():
                if isinstance(module, InputVariationalDropout):
                    module.train()

        # forward `nr_samples` amount of times in batches
        all_outputs = []

        batch_sizes = [self.batch_size] * (self.nr_samples // self.batch_size)
        if self.nr_samples % self.batch_size > 0:
            batch_sizes.append(self.nr_samples % self.batch_size)

        for batch_size in batch_sizes:
            batch = [instance] * batch_size
            outputs = self._model.forward_on_instances(batch)
            all_outputs.extend(outputs)
     

        final_outputs = {
            'words': all_outputs[0]['meta_words']
        }

        for model_key in sub_models.keys():

            # collect class probabilities
            all_class_probs = []
            for outputs in all_outputs:
                tensor = torch.from_numpy(outputs[f'{model_key}_class_probabilities'])
                all_class_probs.append(tensor)
            all_class_probs = torch.stack(all_class_probs)

            print(all_class_probs.shape)

            # calculate mean and variance
            mean = all_class_probs.mean(dim=0)
            std = all_class_probs.std(dim=0)

            final_outputs[f'{model_key}_class_probabilities'] = mean
            final_outputs[f'{model_key}_class_prob_std'] = std

        return sanitize(final_outputs) 


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
