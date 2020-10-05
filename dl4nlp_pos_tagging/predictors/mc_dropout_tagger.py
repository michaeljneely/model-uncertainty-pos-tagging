from overrides import overrides

import torch

from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.data import Instance, DatasetReader
from allennlp.common.util import JsonDict, sanitize
from allennlp.models import Model
from allennlp.modules import InputVariationalDropout


class MCDropoutSentenceTaggerPredictor(SentenceTaggerPredictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader, nr_samples: int = 64, batch_size: int = 32):
        super().__init__(model, dataset_reader)
        self.nr_samples = nr_samples
        self.batch_size = batch_size

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
