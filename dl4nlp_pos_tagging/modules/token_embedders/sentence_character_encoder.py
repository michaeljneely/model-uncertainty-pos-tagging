import torch

from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

import allennlp.nn.util as nn_util

@TokenEmbedder.register("sentence_character_encoding")
class SentenceCharactersEncoder(TokenEmbedder):
    """
    """

    def __init__(self, embedding: Embedding, encoder: Seq2SeqEncoder, dropout: float = 0.0) -> None:
        super().__init__()
        self._embedding = embedding
        self._encoder = encoder
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self) -> int:
        return 2 * self._encoder.get_output_dim()

    def forward(self, token_characters: torch.Tensor) -> torch.Tensor:
        """
        
        token_characters is a tensor of (batch_size, num_tokens, num_characters)

        """
        
        B, S, T = token_characters.shape
        mask = (token_characters != 0).long()                   # B x S x T

        # First we need to merge the S and T dimensions to get sentences (of characters).
        characters = token_characters.view(B, -1)

        # But we need to move the padding at the end of each word to the end of the sentence.
        sentence_indices = torch.arange(S*T, device=characters.device).unsqueeze(0).expand_as(characters)
        to_sort = (characters > 0) * sentence_indices + (S*T+1) * (characters == 0)
        indexer = torch.argsort(to_sort)
        characters = torch.gather(characters, -1, indexer)

        # Embed & Encode
        char_mask = (characters != 0).long()
        embedded = self._embedding(characters)                  # B x S*T x E
        encoded = self._encoder(embedded, char_mask)            # B x S*T x H*directions
 
        token_lengths = mask.sum(-1)
        pad_tokens = token_lengths == 0
        
        # Set length of padding tokens to size of Token dimension.
        token_lengths[pad_tokens] = T
        
        token_lengths = torch.flatten(token_lengths).cumsum(-1)
               
        pad_tokens = pad_tokens.unsqueeze(-1).expand(B, S, T).reshape(B, S*T)
        char_mask = torch.logical_or(char_mask.bool(), pad_tokens.bool())

        very_first_idx = torch.zeros((1), device = encoded.device)
        first_char_idx = torch.cat( (very_first_idx, token_lengths[:-1]), dim=-1).long()
        last_char_idx = token_lengths - 1 

        firsts = encoded[char_mask][first_char_idx]
        lasts = encoded[char_mask][last_char_idx]

        comb = torch.cat((firsts, lasts), dim=-1)
        comb = comb.reshape(B, S, self.get_output_dim())

        return self._dropout(comb)
