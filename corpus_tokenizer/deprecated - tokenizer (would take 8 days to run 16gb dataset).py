from transformers import AutoTokenizer
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from collections import defaultdict
from tqdm import tqdm
import dask.bag as db
import os
import json
import sys

# Training tokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# first thing we do is ge the frequency of the words.
# why? we want each word of course but we also want the freq for the words
# when we go count the number of pairs
# reasoning we don't want to hold and, and, and
# that would take up to much space
# lets use a set ok now we have and
# but we don't know how many times it occurs
# ok lets use dict and map freq to word
# this way when we go and count the number of pairs we know that if 
# we take and we have 3 instances of any pair in and i.e pair an maps to 3
# easy.
# then we can increment that pair again if we see the word andromada 4 times.
# now thet pair an maps to 7.
'''
pre-tokenize text
this will be useful later when we take sentence from corpus.
'''
#     word_freq = defaultdict(int)
#     with open('C:\\Users\\derec\\Desktop\\Fraud classification\\wiki_corpus_files\\wikipedia_corpus.txt', 'r', encoding='utf-8') as file:
#         for line in tqdm(file):
#             if line.strip():
#                 word_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(line)
#                 words = [word for word, _ in word_with_offset]
#                 for word in words:
#                     word_freq[word] += 1


def process_text(text):
    word_freq = defaultdict(int)
    for line in text:
        line = normalizers.BertNormalizer(lowercase=False, strip_accents=True).normalize_str(line)
        if not line.strip():  # Skip empty lines
            continue

        word_with_offset = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(line)
        for word, _ in word_with_offset:
            word_freq[word] += 1
    return list(word_freq.items())

# this brings me back to my questions of why not just t o k e n
# why does it have to be t ##o, ##k, ##e, ##n,
# it makes sense because we can have to so t ##o
# and we can havee motto so ##t ##o these are to different pairs
# why?
# case sensitivity t ##o may occur more then ##t ##o and this tells us
# if we really need to merge it
# we may not want to merge t ##o but we would want to merge  ##t ##o into to
# def compute_pair_scores(splits:dict, word_freq:dict):
#     letter_freq = defaultdict(int)
#     pair_freq = defaultdict(int)

#     for word, freq in word_freq.items():
#         split = splits[word]
#         if len(split) == 1:
#             letter_freq[split[0]] += freq
#             continue
#         for i in range(len(split) - 1):
#             pair = (split[i], split[i + 1])
#             letter_freq[split[i]] += freq
#             pair_freq[pair] += freq
#         letter_freq[split[-1]] += freq

#     scores = {
#         pair: freq / letter_freq[pair[0]] * letter_freq[pair[1]]
#         for pair, freq in pair_freq.items()
#     }

#     return scores   

# def get_bestpair(scores:dict):
#     best_pair = ('','')
#     max_score = 0

#     for pair, score in scores.items():
#         if score > max_score:
#             max_score = score
#             best_pair = pair

#     return best_pair

def get_max_pair(splits:dict, word_freq:dict):
    letter_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    best_pair = ""
    max_score = 0

    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            letter_freq[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freq[split[i]] += freq
            pair_freq[pair] += freq
        letter_freq[split[-1]] += freq

    for pair, freq in pair_freq.items():
        score =  freq / (letter_freq[pair[0]] * letter_freq[pair[1]])
        if score > max_score:
            max_score = score
            best_pair = pair

    return best_pair

def merge_pair(a, b, splits:dict, word_freq:dict):
    for word in word_freq.keys():
        split = splits[word]
        merge = a + b[2:] if b.startswith("##") else a + b
        
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [merge] + split[i + 2:]
            else:
                i += 1

        splits[word] = split

    return merge

def save_progress(token_buffer:list, splits:dict, temp_file:str, final_file:str):
    with open("corpus_tokenizer\\vocab.txt", "a", encoding="utf-8") as file:
        file.write("\n".join(token_buffer) + "\n") # Write all buffered tokens at once
        token_buffer.clear()

    #update files after successfully adding vocab
    #hopefully my ssd isn't cooked after this  :(
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(splits, file, indent=4)

    os.replace(temp_file, final_file)



if __name__ == '__main__':
    folder = "corpus_tokenizer\\"

    '''
    word_freq generation
    '''
    if not os.path.exists(f"{folder}word_freq.txt"):
        b = db.read_text('C:\\Users\\derec\\Desktop\\Fraud classification\\wiki_corpus_files\\wikipedia_corpus.txt', encoding='utf-8', blocksize='256MB')
        word_freqs = b.map_partitions(process_text).compute()

        print('pre-training : get words')
        word_freq = defaultdict(int)
        for word, freq in word_freqs:
            word_freq[word] += freq

        # if it loads and crashes later we don't want to recalc the word_freq
        with open(f"{folder}word_freq.txt", "w", encoding="utf-8") as file:
            for word, freq in tqdm(word_freq.items(), desc='Saving word_freq to word_freq.txt'):
                file.write(f"{word} {freq}\n")
    else:
        #loads it:
        word_freq = {}

        # Load saved progress from the file
        with open(f"{folder}word_freq.txt", "r", encoding="utf-8") as file:
            for line in tqdm(file.readlines(), desc='Loading word_freq'):
                if line.strip():
                    word, freq = line.strip().split()
                    word_freq[word] = int(freq)

    '''
    vocab generation
    '''
    if not os.path.exists(f"{folder}vocab.txt"):
        alphabet = []
        for word in tqdm(word_freq.keys(), desc='pre-training : Initializing Vocab'):
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        alphabet.sort()

        vocab = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]'] + alphabet
    
        with open(f'{folder}vocab.txt', 'w', encoding='utf-8') as file:
            for v in tqdm(vocab, desc='Saving Vocab to vocab.txt'):
                file.write(f"{v}\n")
    else:
        vocab = []

        with open(f'{folder}vocab.txt', 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines(), desc='loading vocab'):
                if line.strip():
                    vocab.append(line.strip())

    # why split all words not first prefixed by ##
    # maps word to their split, i.e  tokenization : [t' ##o, ##k, ##e, ##n, ##i, ##z, ##a, ##t, ##i, ##o, ##n]
    # split is important because we can just look for the word we have and start creating pair
    # if not we could just do it while we process each word, we would make a list.
    # because we need splits later on we make it here.
    # splits = {
    #     word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] 
    #     for word in tqdm(word_freq.keys(), desc='Creating Splits')
    # }

    '''
    save progress of splits as json
    '''
    if not os.path.exists(f'{folder}splits.json'):
        splits = {}
        for word in tqdm(word_freq.keys(), desc='pre-training : Creating Splits'):
            splits[word] = [c if i == 0 else f"##{c}" for i, c in enumerate(word)] 

        print('saving splits to splits.json')
        with open(f"{folder}splits.json", "w", encoding="utf-8") as file:
            json.dump(splits, file, indent=4)
    else:
        # Load JSON file into a dictionary
        print('loading splits from splits.json')
        with open(f"{folder}splits.json", "r", encoding="utf-8") as file:
            splits = json.load(file)

    '''
    Save progress of generating full vocab
    to pick up where it left off.
    '''
    temp_file = f"{folder}splits_temp.json"
    final_file = f"{folder}splits.json"

    token_buffer = []

    pbar = tqdm(total=50000, initial=len(vocab), desc='training : building Vocab')
    vocab_size = 50000
    while len(vocab) < vocab_size:
        best_pair = get_max_pair(splits, word_freq)
        token = merge_pair(best_pair[0], best_pair[1], splits, word_freq)
        if token and token not in vocab:
            vocab.append(token)
            token_buffer.append(token)
            pbar.update(1)

        if len(token_buffer) >= 200:
            save_progress(token_buffer, splits, temp_file, final_file)

            # with open("vocab.txt", "a", encoding="utf-8") as file:
            #     file.write(token + "\n")
    pbar.close()

    if token_buffer:
        save_progress(token_buffer, splits, temp_file, final_file)

    # with open("vocab.txt", "w", encoding='utf-8') as file:
    #     for token in tqdm(vocab, desc='Creating file vocab.txt'):
    #         file.write(token + "\n")

# NOT PART OF TRAINING

# def encode_word(word):
#     tokens = []
#     while len(word) > 0:
#         i = len(word)
#         while i > 0 and word[:i] not in vocab:
#             i -= 1
#         if i == 0:
#             return ["[UNK]"]
#         tokens.append(word[:i])
#         word = word[i:]
#         if len(word) > 0:
#             word = f"##{word}"
#     return tokens

# def tokenize(text):
#     pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
#     pre_tokenized_text = [word for word, _ in pre_tokenize_result]
#     encoded_words = [encode_word(word) for word in pre_tokenized_text]
#     return sum(encoded_words, [])

# print(tokenize("This is the Hugging Face course!"))