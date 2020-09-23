import torch
from overrides import overrides
from typing import List

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

@TokenEmbedder.register("summed-embedding")
class SummedEmbedding(TokenEmbedder):
    """
    This class is meant for summing multiple embeddings together.

    Registered as a `TokenEmbedder` with name "summed-embedding".
    """

    def __init__(self, token_embedders: List[TokenEmbedder]) -> None:
        super().__init__()
        self.token_embedders = token_embedders

        same_dims = all(self.token_embedders[0].get_output_dim() == x.get_output_dim() for x in self.token_embedders)
        if not same_dims:
            raise ValueError("All token embedders must have the same outputs dimensionality.")

        for idx, embedder in enumerate(token_embedders):
            name = "embed_%s" % idx
            self.add_module(name, embedder)

    @overrides
    def get_output_dim(self) -> int:
        return self.token_embedders[0].get_output_dim()

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        
        outputs = None
        for embedder in self.token_embedders:
            embedding = embedder(tokens)
            if outputs == None:
                outputs = embedding
            else:
                outputs += embedding

        return outputs
