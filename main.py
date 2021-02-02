import nltk
import json
import re
import pandas as pd
import pkg_resources

# from nltk.book import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.stem.snowball import SnowballStemmer
from symspellpy import SymSpell
from contractions import CONTRACTION_MAP

corpus_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

sym_spell = SymSpell()




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


# print(replace_negations("they didn't count what he did as an accomplishment."))


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


def penn_to_wn(penntag, return_none=False, default_to_noun=True):
    tag_dict = {
        'NN': wn.NOUN,
        'JJ': wn.ADJ,
        'VB': wn.VERB,
        'RB': wn.ADV
    }
    try:
        return tag_dict[penntag[:2]]
    except:
        if return_none:
            return None
        elif default_to_noun:
            return wn.NOUN
        else:
            return ''

def normalize_sentiment_score(sentiment_sum, tokens_count):
    if tokens_count == 0:
        return 0
    print("\n-------- [INFO]: FINAL SENT SCORE --------\nSentiment: {}, Token Count: {}".format(sentiment_sum, tokens_count))
    sentiment = sentiment_sum / tokens_count
    print("sentiment/tokens_count == {}".format(sentiment))
    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1
    return 0


def sentiment_sentiwordnet(text):
    print("\n-------- [INFO]: SENTIMENT SENTIWORDNET TESTING --------\n")
    raw_sentences = sent_tokenize(text)
    print("\n-------- [INFO]: RAW SENTENCES, TOKENIZE REVIEW TO SENTENCES --------\n{}".format(raw_sentences))

    sentiment_sum = 0
    tokens_count = 0

    for raw_sentence in raw_sentences:

        raw_sentence = expand_contractions(raw_sentence)
        print("\n-------- [INFO]: EXPAND CONTRACTIONS --------\n{}".format(raw_sentence))
        raw_sentence = replace_negations(
            raw_sentence)  # Replacing Negations with Antonyms (Python 3 Text Processing with NLTK 3 Cookbook)
        print("\n-------- [INFO]: REPLACE NEGATIONS --------\n{}".format(raw_sentence))

        tokenized_words = word_tokenize(raw_sentence)
        print("\n-------- [INFO]: TOKENIZE TO WORDS --------\n{}".format(tokenized_words))

        tagged_sentence = pos_tag(tokenized_words)
        print("\n-------- [INFO]: POS TAGGING --------\n{}".format(tagged_sentence))

        tagged_sentence_without_stopwords = [w for w in tagged_sentence if not w[0] in stop_words]
        print("\n-------- [INFO]: REMOVE STOPWORDS --------\n{}".format(tagged_sentence_without_stopwords))

        for word, tag in tagged_sentence_without_stopwords:
            wn_tag = penn_to_wn(tag)

            # print("\n#################### -------- [INFO]: STEMMING RESULT -------- ###################\n{}".format(stemmer.stem(word)))
            # lemma = lemmatizer.lemmatize(stemmer.stem(word), pos=wn_tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                print("******************** THERE IS NO SYNSETS for lemma: ( {} ) ******************\n".format(lemma))
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            word_sent = swn_synset.pos_score() - swn_synset.neg_score()
            print(
                "\n###########-------- [INFO]: LEMMA AND SENTIMENT SCORE OF LEMMA --------###########\nLemma: {}, Sent Score: {}".format(
                    lemma, word_sent))

            if word_sent != 0:
                sentiment_sum += word_sent
                tokens_count += 1

    normalize_sentiment_score(sentiment_sum, tokens_count)
    # if tokens_count == 0:
    #     return 0
    # print("\n-------- [INFO]: FINAL SENT SCORE --------\nSentiment: {}, Token Count: {}".format(sentiment, tokens_count))
    # sentiment = sentiment / tokens_count
    # print("sentiment/tokens_count == {}".format(sentiment))
    #
    # if sentiment >= 0.01:
    #     return 1
    # if sentiment <= -0.01:
    #     return -1
    # return 0


print(sentiment_sentiwordnet("I don't care. He shouldn't have done that. I haven't received it yet. I am not happy. She isn't consistent."))
# print(sentiment_sentiwordnet(data_df["reviewText"][25]))
# print(sentiment_sentiwordnet("I really like the new design of your website! I’m not sure if I like the new design. The new design is awful!"))
# print(sentiment_sentiwordnet("But, as with so many retail stars, GameStop began to struggle a decade or so ago as gamers, like everybody else, made more of their purchases on the internet, opting for downloaded games or two-day delivery over a visit to the mall."))
# print(wn.synset)

# sentiment_sentiwordnet(data_df["reviewText"][1])

# def sentiment_analysis(review_text):

# print(data_df["overall"])