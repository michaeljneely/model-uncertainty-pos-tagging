import torch

from overrides import overrides

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.input_variational_dropout import InputVariationalDropout

@Seq2SeqEncoder.register("dropout")
class Seq2SeqDropout(Seq2SeqEncoder):

    def __init__(self, p: float, input_dim: int, bidirectional: bool = True):
        """
        """
        super().__init__()
        self.dropout = InputVariationalDropout(p = p)
        self.input_dim = input_dim
        self.bidirectional = bidirectional

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        """
        outputs = self.dropout(inputs)

        if mask == None:
            outputs *= mask.unsqueeze(-1)

        return outputs
        
    @overrides
    def get_input_dim(self) -> int:
        return self.input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self.input_dim

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional 


