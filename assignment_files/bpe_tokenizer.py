import argparse
import os
import sys
import multiprocessing
import psutil
import regex as re
import time
import json
from collections import Counter, defaultdict
from typing import BinaryIO
from tqdm import tqdm

def get_mem():
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(t, bytes) for t in split_special_tokens), "All special tokens must be bytes"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    boundaries = [0]
    
    for i in range(1, desired_num_chunks):
        target = i * chunk_size
        file.seek(target)
        
        found = False
        while True:
            curr_pos = file.tell()
            mini_chunk = file.read(4096)
            if not mini_chunk:
                # If we never find a special token, this boundary 
                # effectively moves to the end of the file.
                boundaries.append(file_size)
                break

            found_at = -1
            for token in split_special_tokens:
                idx = mini_chunk.find(token)
                if idx != -1 and (found_at == -1 or idx < found_at):
                    found_at = idx

            if found_at != -1:
                boundaries.append(curr_pos + found_at)
                break
            # If not found, the while loop continues. 
            # file.read(4096) already advanced the pointer.
                
    return sorted(list(set(boundaries)))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize_chunk(
    file_path: str,
    start: int,
    end: int,
    special_tokens_bytes: list[bytes]
) -> Counter:
    """
    Utilizes a regex-based pre-tokenizer from github.com/openai/tiktoken/pull/234/files
    Splits on contractions, words, numbers, symbols, and punctuation 
    Removes special tokens such as <|endoftext|> 
    Returns the frequency of each word in the chunk
    """
    # print(f"Task with boundries {start} - {end}")
    local_counts = Counter()
    with open(file_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start)

    chunk = chunk.replace(b'\r\n', b'\n')
    text = chunk.decode('utf-8', errors='replace')
    
    # Create pattern including the special tokens
    # Ensure splitting on special tokens before the PAT tokens
    if special_tokens_bytes:
        special_pat = '|'.join(re.escape(st.decode('utf-8', errors='replace')) for st in special_tokens_bytes)
        documents = re.split(special_pat, text)
    else:
        documents = [text]
    
    compiled_pat = re.compile(PAT)

    for doc in documents:
        if not doc:
            continue
        for match in compiled_pat.finditer(doc):
            token_bytes = match.group().encode('utf-8')
            local_counts[token_bytes] += 1

    return local_counts


def get_token_counts(
        file_path: str,
        procs: int,
        special_tokens:list[str]
    ) -> Counter:
    
    # Arguments
    print('Dataset: ', file_path)
    print('Number of processes requested: ', procs)
    special_tokens_bytes = [t.encode('utf-8') for t in special_tokens]
    print('Special tokens as bytes: ', special_tokens_bytes)

    # chunking
    with open(file_path, 'rb') as f:
        if special_tokens:
            boundaries = find_chunk_boundaries(f, procs, special_tokens_bytes)
        else:
            f.seek(0, os.SEEK_END)
            boundaries = [0, f.tell()]

    # Parallelize chunking
    tasks = []
    for i in range(len(boundaries) - 1):
        tasks.append((file_path, boundaries[i], boundaries[i+1], special_tokens_bytes))
    
    print('Total number of tasks to complete: ', len(tasks))
    num_workers = min(len(tasks), procs)
    with multiprocessing.Pool(processes=num_workers) as pool:
        # starmap due to how the parameters are accepted in pre_tokenize_chunk
        all_counts = pool.starmap(pre_tokenize_chunk, tasks)
    
    # aggregate
    agg_counts = Counter()
    for res in all_counts:
        agg_counts.update(res)

    return agg_counts

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    input_path: path to an input text file
    vocab_size: maximum final vocabulary size 
                (merges and including the initial byte vocabulary,
                vocabulary items produced from merging, and any special tokens)
    special_tokens: list of strings to add to the vocabulary that do not affect BPE training

    returns  vocabulary and merges
    vocab: tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    merges: A list of BPE merges produced from training ordered by creation order
    """
    # measurements
    start_time = time.perf_counter()
    start_mem = get_mem()

    # Pretokenize and Count tokens
    procs = multiprocessing.cpu_count()
    token_counts = get_token_counts(input_path, procs, special_tokens)
    pre_tokenize_time = time.perf_counter()

    print(f"Pre-tokenization complete. Unique pre-tokens: {len(token_counts)}")

    # encode each character in each word obtained by get_token_counts()
    # key: tuple containing the utf-8 encoding of each character in a word
    # value: the frequency of the word 
    word_freqs = {tuple(bytes([b]) for b in word): freq 
                  for word, freq in token_counts.items()}

    # initialize 
    vocab = {x: bytes([x]) for x in range(256)}
    merges = []
    current_size = len(vocab)

    num_merges = vocab_size - 256 - len(special_tokens)
    print('Number of merges to perform: ', num_merges)
    
    pair_freqs = Counter() # Counts the occurances of every pair
    pair_to_words = defaultdict(set) # Associates pairs to a set of words that contain the pair


    # word_freqs = {word byte encoding: frequency}
    # For each word create and record the amount of pairs
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            # forms a pair 
            # record or update the number of seen pairs 
            # adds the word to a set of words with that pairing 
            pair = (word_tuple[i], word_tuple[i+1]) 
            pair_freqs[pair] += freq 
            pair_to_words[pair].add(word_tuple) 

    initialization_time = time.perf_counter()

    # Begin merging
    for i in tqdm(range(num_merges), desc="Merging Vocab"):
        # print('Current Vocab size ', current_size)
        if not pair_freqs:
            print(f"No more pairs to merge after {i} merges.")
            break

        # Find the max frequency pair(s)
        # In the case of a tie select lexicographically greater pair
        max_freq = max(pair_freqs.values())
        candidates = [p for p, f in pair_freqs.items() if f == max_freq]
        # if len(candidates) > 1:
        #     print(f"\n--- Tie detected at Freq: {max_freq} ---")
        #     print(f"Candidates: {candidates}")
        #     print(f"Selected (max): {max(candidates)}")
        #     print(f"Selected (min): {min(candidates)}")
        max_pair = max(candidates)

        # Merge the two bytes/ tokens, add it to the vocab and the merge list 
        new_token = max_pair[0] + max_pair[1]
        vocab[current_size] = new_token
        merges.append(max_pair)
        current_size += 1

        # Get list of words from pair_to_words to update with the new pair 
        affected_words = list(pair_to_words[max_pair])
        for word in affected_words:
            # remove the word from word_freqs
            freq = word_freqs.pop(word)

            # remove all pairs of the word
            for j in range(len(word) - 1):
                p = (word[j], word[j+1]) # create the pair
                pair_freqs[p] -= freq # delete the frequency of that pair
                pair_to_words[p].discard(word) # remove the word for that pair
                # if all counts of the pair is gone, remove that pair
                if pair_freqs[p] <= 0:
                    del pair_freqs[p]

            # Recreate the word with the new merge, and re-add previous merges
            new_word = []
            k = 0
            while k < len(word):
                if k < len(word) - 1 and (word[k], word[k+1]) == max_pair:
                    new_word.append(new_token)
                    k += 2
                else:
                    new_word.append(word[k])
                    k += 1

            # Re-add the byte encoding: frequency to word_freqs
            new_word_tuple = tuple(new_word)
            word_freqs[new_word_tuple] = freq

            # Re-add frequency of pairs to pair_freqs
            # Re-add words to each pair and the new merge
            for m in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[m], new_word_tuple[m+1])
                pair_freqs[p] += freq
                pair_to_words[p].add(new_word_tuple)

     # Add special tokens
    for special in special_tokens:
        assert isinstance(special, str)
        spe_enc = special.encode('utf-8')
        if spe_enc not in vocab.values():
            vocab[current_size] = spe_enc
            current_size += 1

    merge_time = time.perf_counter()
    end_mem = get_mem()
    print(f"Pre-tokenize time complete in {pre_tokenize_time - start_time:.2f}s")
    print(f"Initialization time complete in {initialization_time - pre_tokenize_time:.2f}s")
    print(f"BPE Merges complete in {merge_time - initialization_time:.2f}s")
    print(f"Total training time: {merge_time - start_time:.2f}s")
    print(f"Total memory used: {end_mem - start_mem:.2f} MB")
    return vocab, merges
    #TODO

def save_vocab(vocab, file_path):
    # Convert {0: b'a'} to {0: [97]} for JSON safety
    serializable_vocab = {
        token_id: list(token_bytes) 
        for token_id, token_bytes in vocab.items()
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, indent=4)
    print(f"Vocab saved to {file_path}")

def save_merges(merges, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        
        for pair in merges:
            # We convert the bytes to hex strings (e.g., '61 62') 
            # so the text file is readable and won't break on special chars
            left = pair[0].hex()
            right = pair[1].hex()
            f.write(f"{left} {right}\n")
    print(f"Merges saved to {file_path}")

# Test run with
# python .\bpe_tokenizer.py ..\..\data\owt_valid.txt
# python .\bpe_tokenizer.py ..\..\data\owt_train.txt

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Pre-tokenizer')
    parser.add_argument('file', type=str, help='Path to the dataset')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    voc, mer = train_bpe(args.file, 10000, ['<|endoftext|>'])
    save_vocab(voc, 'testvocab.json')
    save_merges(mer, 'testmerge.json')
    # save_vocab(voc, 'TinyStoriesVocab.json')
    # save_merges(mer, 'TinyStoriesMerges.json')

