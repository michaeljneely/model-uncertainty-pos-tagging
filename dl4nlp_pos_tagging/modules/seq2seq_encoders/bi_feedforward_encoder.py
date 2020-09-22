from overrides import overrides

from allennlp.modules.seq2seq_encoders.feedforward_encoder import FeedForwardEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("bi-feedforward")
class BiFeedForwardEncoder(FeedForwardEncoder):

    @overrides
    def is_bidirectional(self) -> bool:
        return True 

