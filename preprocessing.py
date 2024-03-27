# Read the hp_books.txt file
with open('hp-gen\hp_books.txt', 'r') as file:
    hp_books_content = file.read()

# Open the english_words.txt file
with open('hp-gen\eng_words.txt', 'r') as file:
    english_words_content = file.read()

english_words = set(english_words_content.split())
hp_words = set(hp_books_content.split())


# Remove words from hp_words that exist in english_words
hp_words = hp_words - english_words



print(hp_words)