from typing import Iterable
import itertools 
import logging

from allennlp_models.tagging.dataset_readers import Conll2000DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

def _is_divider(line: str) -> bool:
    return line.strip() == ""

logger = logging.getLogger(__name__)

@DatasetReader.register("spaced-conll2000")
class SpacedConll2000DatasetReader(Conll2000DatasetReader):


    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        This version is identical to the original in AllenNLP, but
        adds spaces to the tokens.
        """
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags = fields
   
                    # Add spaces to each token
                    for i, token in enumerate(tokens_[:-1]): 
                        tokens_[i] = token + " "

                    # TextField requires `Token` objects
                    tokens = [Token(token) for token in tokens_]

                    yield self.text_to_instance(tokens, pos_tags, chunk_tags)
