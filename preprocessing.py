import spacy
from pathlib import Path
from tqdm import tqdm
import re
"""
We boil down the Harry Potter books to a list of unique lemmas (dictionary form of a word).
Then we compare them to an english dictionary to extract words that only exist in the Harry Potter books.
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


def read_text(path: Path) -> str:
    with open(path, 'r', encoding='utf8') as file:
        content = file.read()
    return content


def normalize_and_split_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"[^a-z']", " ", text)
    text = re.sub(r"'", " '", text) # we want to keep things like 've, 'll, etc.
    return text.split()


def first_filter(words_minuend: list[str], words_subtractor: list[str]) -> list[str]:
    words_minuend = set(words_minuend)
    words_subtractor = set(words_subtractor)
    words_difference = words_minuend - words_subtractor
    return words_difference


def get_unique_lemmas_from_words(words: set[str], nlp) -> set[str]:    
    lemmas = []
    print(f'Lemmatizing {len(words)} words...\n')
    for word in tqdm(words):
        doc = nlp(word)
        lemmas.extend([token.lemma_ for token in doc])
      
    unique_lemmas = set(lemmas)
    print(f'Complete. There are {len(unique_lemmas)} unique lemmas in the text.\n')    
    return unique_lemmas


if __name__ == '__main__':    
    hp_text = read_text('hp_books.txt')
    hp_normalized = normalize_and_split_text(hp_text)
    print(hp_normalized[:40])
    english_text =read_text('eng_words.txt')
    english_normalized = normalize_and_split_text(english_text)
    print(english_normalized[:10])
    
    hp_first_filter = first_filter(hp_normalized, english_normalized)
    print(len(hp_first_filter))

    spacy_model = 'en_core_web_sm'
    nlp = load_spacy_model(model_name=spacy_model)
    hp_lemmas = get_unique_lemmas_from_words(hp_first_filter, nlp=nlp)

    english_words = set(english_normalized)
    hp_only_words = hp_lemmas - english_words
    
    output_path = Path('hp_style_words.txt')
    with open(output_path, "w") as file:
        file.write("\n".join(hp_only_words))
    print(f'Wrote {len(hp_only_words)} unique words to {output_path}')