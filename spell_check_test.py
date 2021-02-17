import nltk.corpus
import pkg_resources

import re
from symspellpy import SymSpell, Verbosity

corpus_words = set(nltk.corpus.words.words())

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")

# term_index is the column of the term and count_index is the
# column of the term frequency
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


# input_term = "you can stdy the art in detil and it als shws how a divrse group of artists worked together to crete someting unque and memorable that came out of one persns vison"
#
# print("original: ", input_term)
# top_suggestion = spelling_correction(input_term)
# # suggestion = suggestions[0].term
# print("suggestion: ", top_suggestion)
# # i am already a baseball fan and knew a bit about the negro leagues but i learned a lot more reading this book
