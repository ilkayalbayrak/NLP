import nltk.corpus
import pkg_resources

import re
from symspellpy import SymSpell, Verbosity

corpus_words = set(nltk.corpus.words.words())

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def word_segmentation(text):
    return None


def find_top_spelling_suggestion(text):
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
    return suggestions[0].term


def spelling_correction(text):
    if isinstance(text, str):
        text = re.findall(r'\w+', text)
        checked_word_list = []

        for word in text:
            if not word in corpus_words:
                checked_word_list.append(find_top_spelling_suggestion(word))
            else:
                # print("else")
                checked_word_list.append(word)
        return " ".join(checked_word_list)

    elif isinstance(text, list):
        checked_word_list = []

        for word in text:
            if not word in corpus_words:
                checked_word_list.append(find_top_spelling_suggestion(word))
            else:
                # print("else")
                checked_word_list.append(word)
        return " ".join(checked_word_list)


input_term = "i am alredy a bseball fan and knw a bit abot the african legues but i learnd a lot more reding this bok"

print("original: ", input_term)
top_suggestion = spelling_correction(input_term)
# suggestion = suggestions[0].term
print("suggestion: ", top_suggestion)
