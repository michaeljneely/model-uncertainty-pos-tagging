from overrides import overrides

from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor


class MCDropoutSentenceTaggerPredictor(SentenceTaggerPredictor):


    def __init__(self, nr_samples: int):
        super().__init__()
        self.nr_samples = nr_samples


    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:

        for s in range(self.nr_samples):
            `
