from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary

@Model.register("meta-model")
class MetaModel(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        sub_models: List[Model],
        meta_model: Model
        training: bool = True
    ):
        """
        TODO:
        """
        super().__init__(vocab)
        self.word_model = word_model
        self.character_model = character_model
        self.meta_model = meta_model

        self.models = [word_model, character_model, meta_model]

    def forward(self, ) -> Dict[str, torch.Tensor]:


        if self.training:
            # switch between models based on current epoch.

            if self.epoch == None:
                pass # TODO warning

            if self.epoch 

            return None

        

        

