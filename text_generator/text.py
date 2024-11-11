import nltk
import random
import re
from nltk import ngrams
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

class CreativeTextGenerator:
    def __init__(self, n=3):
        self.n = n  # N-gram size
        self.model = {}

    def train_model(self, text):
        tokens = word_tokenize(text.lower())
        n_grams = list(ngrams(tokens, self.n, pad_left=True, pad_right=True))

        for i in range(len(n_grams) - 1):
            gram = n_grams[i]
            next_word = n_grams[i + 1][-1]

            if gram not in self.model:
                self.model[gram] = []

            self.model[gram].append(next_word)

    def generate_text(self, seed_text, max_length=50):
        """
        Generate text based on seed_text using the trained N-gram model.
        """
        tokens = word_tokenize(seed_text.lower())
        current_gram = tuple(tokens[-self.n:])
        output = list(current_gram)

        for _ in range(max_length):
            if current_gram not in self.model:
                break

            possible_words = self.model[current_gram]
            next_word = random.choice(possible_words)
            output.append(next_word)

            current_gram = tuple(output[-self.n:])

        return ' '.join(output)

class TextEnhancer:
    def __init__(self):
        pass

    def synonym_replacement(self, text):
        tokens = word_tokenize(text)
        enhanced_text = []

        for word in tokens:
            synonyms = self.get_synonyms(word)
            if synonyms:
                enhanced_text.append(random.choice(synonyms))
            else:
                enhanced_text.append(word)

        return ' '.join(enhanced_text)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)

    def grammar_correction(self, text):
        corrected_text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)  # Remove spaces before punctuation
        corrected_text = re.sub(r'([?.!,"])\s+', r'\1 ', corrected_text)  # Ensure a space after punctuation
        return corrected_text

def main():
    choice = input("Do you want to load training text from a file? (yes/no):\n").lower()
    if choice in ['yes', 'y']:
        file_path = input("Enter the file path:\n")
        with open(file_path, 'r', encoding='utf-8') as file:
            training_text = file.read()
    else:
        training_text = input("Enter the training text (or leave blank for a default example):\n")
        if not training_text:
            training_text = """
            The quick brown fox jumps over the lazy dog. The dog barks at the moon. 
            Foxes are clever animals known for their cunning nature. 
            A fox often jumps swiftly to catch its prey. 
            """

    generator = CreativeTextGenerator(n=3)
    enhancer = TextEnhancer()
    generator.train_model(training_text)

    seed_text = input("\nEnter the seed text for generating text:\n")
    if not seed_text:
        seed_text = "The quick brown"

    max_length = input("\nEnter the maximum number of words to generate (default is 20):\n")
    max_length = int(max_length) if max_length.isdigit() else 20

    generated_text = generator.generate_text(seed_text, max_length=max_length)
    print("\nGenerated Text:\n", generated_text)

    enhance_choice = input("\nDo you want to enhance the text with synonym replacement? (yes/no):\n").lower()
    if enhance_choice in ['yes', 'y']:
        enhanced_text = enhancer.synonym_replacement(generated_text)
        print("\nEnhanced Text (Synonym Replacement):\n", enhanced_text)

        corrected_text = enhancer.grammar_correction(enhanced_text)
        print("\nCorrected Text (Grammar Correction):\n", corrected_text)
    else:
        print("\nText enhancement skipped.")

if __name__ == "__main__":
    main()
