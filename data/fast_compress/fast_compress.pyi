from typing import List, Optional, Tuple

def compress(
    ids: List[int],
    initial_vocab_size: int,
    max_codebook_size: int,
    max_out_seq_length: int,
    max_subtokens: int,
    eot_token_id: int,
    disabled_ids: Optional[List[int]],
) -> Tuple[List[int], List[List[int]], Optional[List[int]]]: ...
