import spacy
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import Levenshtein
"""
We compare all words from the Harry Potter books to an english dictionary to extract words that only exist in the books.
"""

def load_spacy_model(model_name: str) -> spacy.language.Language:
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f'''Downloading language model {model_name } for the spaCy POS tagger\n
            (don't worry, this will only happen once)''')
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def read_normalize_split_text(path: Path) -> list[str]:
    with open(path, 'r', encoding='utf8') as file:
        content = file.read()
    content = _normalize_and_split_text(content)
    return content


def _normalize_and_split_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"'", " '", text) # we want to keep things like 've, 'll, etc.
    return text.split()


def remove_equal_words(words_minuend: list[str], words_subtractor: list[str]) -> list[str]:
    words_minuend = set(words_minuend)
    words_subtractor = set(words_subtractor)
    words_difference = words_minuend - words_subtractor
    return list(words_difference)


def get_unique_lemmas_from_words(words: list[str], nlp) -> list[str]:
    words = set(words)
    lemmas = []
    print(f'Lemmatizing {len(words)} words...')
    for word in tqdm(words):
        doc = nlp(word)
        lemmas.extend([token.lemma_ for token in doc])
      
    unique_lemmas = list(set(lemmas))
    print(f'Complete.\n')    
    return unique_lemmas


def remove_similar_words(words: list[str], dictionary: list[str], similarity_limit) -> list[str]:
    not_similar_words = []
    print(f'Comparing {len(words)} words to {len(dictionary)} words in the dictionary...')
    for word in tqdm(words):
        if not any(_calculate_character_similarity(word, eng_word) > similarity_limit for eng_word in dictionary):
            not_similar_words.append(word)
    print(f'Complete.\n')
    return not_similar_words


def _calculate_character_similarity(word1: str, word2: str) -> float:
    levenshtein_distance = Levenshtein.distance(word1, word2)
    similarity = 1 - levenshtein_distance / max(len(word1), len(word2))
    return similarity


if __name__ == '__main__':
    nouns = read_normalize_split_text('eng_dictionary/nouns.txt')
    verbs = read_normalize_split_text('eng_dictionary/verbs.txt')
    adjectives = read_normalize_split_text('eng_dictionary/adjectives.txt')
    adverbs = read_normalize_split_text('eng_dictionary/adverbs.txt')
    general_words = read_normalize_split_text('eng_dictionary/words.txt')
    english_dictionary = set(nouns + verbs + adjectives + adverbs + general_words)
    print(f'Loaded {len(english_dictionary)} words from the English dictionary.\n')
    
    hp_words = read_normalize_split_text('hp_books_faulty.txt')
    print(f'Starting preprocessing with {len(hp_words)} words from the books...\n')
    
    equal_filter = remove_equal_words(hp_words, english_dictionary)
    print(f'After checking matches with the dictionary, {len(equal_filter)} words remain.\n')
    
    hp_lemmas = get_unique_lemmas_from_words(equal_filter, load_spacy_model('en_core_web_sm'))
    lemma_filter = remove_equal_words(hp_lemmas, english_dictionary)
    print(f'After lemmatizing and comparing, {len(lemma_filter)} words remain.\n')
    
    similarity_limit = 0.66
    similarity_filter = remove_similar_words(lemma_filter, english_dictionary, similarity_limit=similarity_limit)
    print(f'After removing words with more than {similarity_limit * 100}% similarity, {len(similarity_filter)} words remain.\n')

    output_path = Path('hp_style_words.txt')
    with open(output_path, "w") as file:
        file.write("\n".join(similarity_filter))
    print(f'Wrote {len(similarity_filter)} unique words to {output_path}')
