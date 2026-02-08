import json
import regex as re
import os
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # Reverse lookup for encoding {bytes, id}
        self.vocab_reversed = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # Create special pattern, use the longer special pattern first
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_re_str = "|".join(re.escape(t) for t in sorted_specials)
            self.special_pat = re.compile(f"({special_re_str})")
        else:
            self.special_pat = None


    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens=None
    ):
        # Construct vocab
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab = {int(k): bytes(v) for k, v in data.items()}
        
        if special_tokens:
            next_id = max(vocab.keys()) + 1 if vocab else 0
            existing_byte_values = set(vocab.values())
            for st in special_tokens:
                st_bytes = st.encode('utf-8')
                if st_bytes not in existing_byte_values:
                    # Assign the next available integer ID
                    existing_byte_values.add(st_bytes)
                    vocab[next_id] = st_bytes
                    next_id += 1

        # Construct merges
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        merge_tuple = (bytes.fromhex(parts[0]), bytes.fromhex(parts[1]))
                        merges.append(merge_tuple)
                    except ValueError:
                        continue
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Processes chunks of text one by one, yielding tokens as they are found
        to keep memory usage below the 1MB limit.
        """
        # print('encode_itr run')
        for chunk in iterable:
            if not chunk: continue
            
            if self.special_pat:
                last_pos = 0
                for match in self.special_pat.finditer(chunk):
                    pre_text = chunk[last_pos:match.start()]
                    if pre_text:
                        yield from self._yield_regular_tokens(pre_text)
                    
                    special_token = match.group()
                    yield self.vocab_reversed[special_token.encode('utf-8')]
                    last_pos = match.end()
                
                remaining = chunk[last_pos:]
                if remaining:
                    yield from self._yield_regular_tokens(remaining)
            else:
                yield from self._yield_regular_tokens(chunk)

    def _yield_regular_tokens(self, text: str) -> Iterator[int]:
        for match in self.PAT.finditer(text):
            chunk_bytes = match.group().encode('utf-8')
            yield from self._yield_merged_tokens(chunk_bytes)
                
    def _yield_merged_tokens(self, word_bytes: bytes) -> Iterator[int]:
        """Faster BPE merging using rank lookups."""
        if word_bytes in self.vocab_reversed:
            yield self.vocab_reversed[word_bytes]
            return
    
        parts = [bytes([b]) for b in word_bytes]
    
        while len(parts) > 1:
            # Find all possible byte pairs
            pairs = [(parts[i], parts[i+1]) for i in range(len(parts) - 1)]
            
            # Find which of these pairs has the best (earliest) merge
            # If a pair isn't in merge_ranks, we give it 'inf'
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
    
            if best_pair not in self.merge_ranks:
                break
    
            # Perform the merge
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
    
        for p in parts:
            yield self.vocab_reversed[p]

    def decode(self, ids: list[int]) -> str:
        # List of bytes to avoid frequent string concatenation
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode('utf-8', errors='replace')