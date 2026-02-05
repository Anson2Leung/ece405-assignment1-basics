import json
import os

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # JSON keys are always strings, so we convert "0" -> 0
        serializable_vocab = json.load(f)
    
    # Reconstruct the original {int: bytes} format
    vocab = {
        int(token_id): bytes(token_list) 
        for token_id, token_list in serializable_vocab.items()
    }
    return vocab

# vocab = load_vocab("ece405-assignment1-basics/assignment_files/OWTVocab.json")
vocab = load_vocab("ece405-assignment1-basics/assignment_files/TinyStoriesVocab.json")

def get_decoded_string(token_bytes):
    return token_bytes.decode('utf-8', errors='ignore')

longest_token_id = max(vocab, key=lambda k: len(get_decoded_string(vocab[k])))
longest_token_bytes = vocab[longest_token_id]
longest_text = get_decoded_string(longest_token_bytes)

print(f"--- Analysis Results ---")
print(f"Longest Token ID: {longest_token_id}")
print(f"Longest Token Text: '{longest_text}'")
print(f"Character Length: {len(longest_text)}")