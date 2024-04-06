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
    text = re.sub(r'P a g e |', '', text)
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r'\b\w*?(.)\1{2,}\w*\b', '', text)  # removes words with 3 or more consecutive equal characters (aaargh, etc.)
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"'", " '", text) # we want to keep things like 've, 'll, etc.
    return text.split()


def _remove_equal_words(words_minuend: list[str], words_subtractor: list[str]) -> list[str]:
    words_minuend = set(words_minuend)
    words_subtractor = set(words_subtractor)
    words_difference = words_minuend - words_subtractor
    return list(words_difference)


def _get_unique_lemmas_from_words(words: list[str], nlp) -> list[str]:
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
        if not any(_calculate_character_similarity(word, dictionary_word) > similarity_limit for dictionary_word in dictionary):
            not_similar_words.append(word)
    print(f'Complete.\n')
    return not_similar_words


def _calculate_character_similarity(word1: str, word2: str) -> float:
    levenshtein_distance = Levenshtein.distance(word1, word2)
    similarity = 1 - levenshtein_distance / max(len(word1), len(word2))
    return similarity


def combine_text_files(directory: Path) -> None:
    combined_dictionary = set()
    for dictionary_file in directory.glob('*.txt'):
        with open(dictionary_file, 'r', encoding="utf8") as file:
            content = file.read()
        content = _normalize_and_split_text(content)
        combined_dictionary.update(content)
 
    with open(directory / "combined.txt", 'w+') as file:
        file.write("\n".join(combined_dictionary))


def run_cleaning_pipeline(book: set[str], dictionary: set[str], output_path: Path, similarity_limit: float | None = None) -> None:    
    equal_filter = _remove_equal_words(book, dictionary)
    print(f'After checking matches, {len(equal_filter)} words remain.\n')
    
    lemmas = _get_unique_lemmas_from_words(equal_filter, load_spacy_model('en_core_web_sm'))
    lemma_filter = _remove_equal_words(lemmas, dictionary)
    print(f'After lemmatizing, {len(lemma_filter)} words remain.\n')
    
    if similarity_limit:
        similarity_filter = remove_similar_words(lemma_filter, dictionary, similarity_limit)
        print(f'After removing words with more than {similarity_limit * 100}% similarity, {len(similarity_filter)} words remain.\n')
        with open(output_path, "w") as file:
                file.write("\n".join(similarity_filter))
        print(f'Wrote {len(similarity_filter)} unique words to {output_path}')
    else:
        print('No similarity limit provided, Trying a few...\nThis may take a while.')
        for limit in [0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.72, 0.74, 0.76, 0.78, 0.8]:
            similarity_filter = remove_similar_words(lemma_filter, dictionary, similarity_limit=limit)
            print(f'After removing words with more than {limit * 100}% similarity, {len(similarity_filter)} words remain.\n')
            
            test_path = Path(f'similarity_limit_{limit}.txt')
            with open(test_path, "w") as file:
                file.write("\n".join(similarity_filter))
            print(f'Wrote {len(similarity_filter)} unique words to {test_path}')


if __name__ == '__main__':
    dictionary_path = Path('eng_dictionary')
    books_path = Path('hp_series')
    
    # Read dictionary
    if not Path(dictionary_path / 'combined.txt').exists():
        combine_text_files(directory=Path(dictionary_path))
    dictionary = read_normalize_split_text(dictionary_path / 'combined.txt')
    print(f'Loaded {len(dictionary)} words from the dictionary.\n')
    
    # Read books
    if not Path(books_path / 'combined.txt').exists():
        combine_text_files(directory=Path(books_path))
    books = read_normalize_split_text(books_path / 'combined.txt')
    print(f'Loaded {len(books)} words from the books.\n')
    
    run_cleaning_pipeline(books, dictionary, output_path='hp_style_words.txt', similarity_limit=0.68)
    
    
    
    
    
