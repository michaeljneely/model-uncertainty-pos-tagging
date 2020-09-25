import torch

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TokenEmbedder.register("sentence_character_encoding")
class SentenceCharactersEncoder(TokenEmbedder):
    """
    """

    def __init__(self, embedding: Embedding, encoder: Seq2SeqEncoder, dropout: float = 0.0) -> None:
        super().__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return self._encoder._module.get_output_dim()

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        """
        
        token_characters is a tensor of (batch_size, num_tokens, num_characters)

        """
        mask = (token_characters != 0).long()
        token_lengths = mask.sum(-1)
        seq_lengths = token_lengths.sum(-1)

        embedded = self._embedding(token_characters) # B x T x C x D

        # pack sequence so we have all characters in one 
        packed = torch.nn.utils.rnn.pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False)
        char_sequence = packed.data
    
        self._encoder(
