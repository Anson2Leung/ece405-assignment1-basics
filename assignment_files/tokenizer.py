import json
import re
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
        # Reverse lookup for encoding
        self.vocab_reversed = {v: k for k, v in vocab.items()}
        
        # Add special tokens to vocab
        if special_tokens:
            for token in special_tokens:
                t_bytes = token.encode('utf-8')
                if t_bytes not in self.vocab_reversed:
                    new_id = max(self.vocab.keys(), default=-1) + 1
                    self.vocab[new_id] = t_bytes
                    self.vocab_reversed[t_bytes] = new_id

        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens=None
    ):
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab = {int(k): bytes(v) for k, v in data.items()}
        
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    l_hex, r_hex = line.strip().split()
                    merges.append((bytes.fromhex(l_hex), bytes.fromhex(r_hex)))
        
        return cls(vocab, merges, special_tokens)

    def _merge_pre_token(self, word_bytes: bytes) -> list[int]:
        parts = [bytes([b]) for b in word_bytes]
        
        # Apply learned merges in order of creation
        for pair_l, pair_r in self.merges:
            target = pair_l + pair_r
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair_l and parts[i+1] == pair_r:
                    new_parts.append(target)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
            
        # Convert the final merged byte sequences to IDs
        return [self.vocab_reversed[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        ids = []
        # Pre-tokenize (handles special tokens if included in text)
        for match in self.pat.finditer(text):
            pre_token_bytes = match.group().encode('utf-8')
            ids.extend(self._merge_pre_token(pre_token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # By processing line-by-line or chunk-by-chunk, 
            # we ensure constant memory complexity.
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        # Use a list of bytes to avoid frequent string concatenation
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode('utf-8', errors='replace')