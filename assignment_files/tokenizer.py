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

    def _merge_pre_token(self, word_bytes: bytes) -> list[int]:
        parts = [bytes([b]) for b in word_bytes]
        
        # Merge if the left half and right half exist in merges
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
        if not text:
            return []
        if self.special_pat is None:
            return self._encode_regular_text(text)

        
        ids = []
        # Split the line on special tokens
        parts = self.special_pat.split(text)
        for i, part in enumerate(parts):
            # special tokens will be on odd indicies separated by text or ""
            if i % 2 == 1:
                # This is a confirmed special token
                ids.append(self.vocab_reversed[part.encode('utf-8')])
            elif part:
                # This is regular text
                ids.extend(self._encode_regular_text(part))
        return ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def _encode_regular_text(self, text: str) -> list[int]:
        ids = []
        for match in self.PAT.finditer(text):
            chunk_bytes = match.group().encode('utf-8')
            ids.extend(self._merge_pre_token(chunk_bytes))
        return ids

    def decode(self, ids: list[int]) -> str:
        # Use a list of bytes to avoid frequent string concatenation
        raw_bytes = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return raw_bytes.decode('utf-8', errors='replace')