import nltk
import json
import re
import pandas as pd

# from nltk.book import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.stem.snowball import SnowballStemmer
from contractions import CONTRACTION_MAP

corpus_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')



# print(nltk.__version__)
# print(pd.__version__)


# data = pd.read_json("data/All_Beauty.json", orient="index", lines=True)
# print(data)

def parse(path):
    with open(path) as file:
        for line in file:
            yield json.loads(line)


def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


pd.set_option('display.max_columns', None)

# pd.set_option('display.expand_frame_repr', False)
path = "data/test_data.json"

# ratings = []

# for review in parse(path):
#     ratings.append(review['overall'])

# print(sum(ratings) / len(ratings))
#
data_df = get_df(path)


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# for i in tokens:
#     lem_tokens.append(lemmatizer.lemmatize(i))
# print(lem_tokens)

def replace(word, pos=None):
    antonyms = set()

    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())

    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None


def replace_negations(text):
    text = re.findall(r'\w+', text)
    i, j = 0, len(text)
    words = []

    while i < j:
        word = text[i]

        if word == 'not' and i + 1 < j:
            ant = replace(text[i + 1])

            if ant:
                words.append(ant)
                i += 2
                continue

        words.append(word)
        i += 1

    return ' '.join(words)


print(replace_negations("they didn't count what he did as an accomplishment."))


# print("\n-------- [INFO]: DF.info --------\n{}".format(data_df.info()))
# print("\n-------- [INFO]: DF.describe --------\n{}".format(data_df.describe()))
#
# print("\n-------- [INFO]: DF.columns --------\n{}".format([x for x in data_df.columns.values]))
#
# print("\n-------- [INFO]: DF.isnull --------\n{}".format(data_df.isnull().sum()))
# # print(data_df.head())
#
#
# tokens = nltk.word_tokenize(data_df["reviewText"][1])
# print("\n-------- [INFO]: RAW TEXT --------\n{}".format(data_df["reviewText"][1]))


# print("\n-------- [INFO]: TOKENS --------\n{}".format(tokens))
#
# lemmatizer = WordNetLemmatizer()
# lem_test = lemmatizer.lemmatize("misspelled", pos="v")
# print("\n-------- [INFO]: LEMMA TEST --------\n{}".format(lem_test))
# lem_tokens = []


# def clean_punctuation:


# ref_text = expand_contractions("Y'all can't expand contractions I'd think\nI really don't understand how this isn't possible, I haven't been trying hard though")
# ref_text = expand_contractions(data_df["reviewText"][1])
# print("\n-------- [INFO]: REPLACE CONTRACTIONS --------\n{}".format(ref_text))



