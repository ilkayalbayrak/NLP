import nltk
import json
import re
import pandas as pd
# import pkg_resources ## for symspell

# from nltk.book import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.stem.snowball import SnowballStemmer
# from symspellpy import SymSpell
from contractions import CONTRACTION_MAP
from time import process_time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

corpus_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


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


def filter_text(text):
    # Filter only letters and spaces in the text
    pattern = re.compile(r'[^a-zA-Z ]+')
    letters_only = re.sub(pattern, '', str(text))
    return letters_only


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


def normalize_sentiment_score(sentiment_sum, tokens_count):
    if tokens_count == 0:
        return 0
    # print("\n-------- [INFO]: FINAL SENT SCORE --------\nSentiment: {}, Token Count: {}".format(sentiment_sum,
    #                                                                                             tokens_count))
    sentiment = sentiment_sum / tokens_count
    # print("sentiment/tokens_count == {}".format(sentiment))
    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1
    return 0


def normalize_ratings(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return -1
    else:
        return 0


def sentiment_analyzer(text):
    # print("\n-------- [INFO]: SENTIMENT SENTIWORDNET TESTING --------\n")
    original_sentences = sent_tokenize(text)
    # print("\n-------- [INFO]: RAW SENTENCES, TOKENIZE REVIEW TO SENTENCES --------\n{}".format(original_sentences))

    sentiment_sum = 0
    tokens_count = 0

    for sentence in original_sentences:

        sentence = expand_contractions(sentence)
        # print("\n-------- [INFO]: EXPAND CONTRACTIONS --------\n{}".format(sentence))
        # REMOVE EVERYTHING BUT LETTERS
        sentence = filter_text(sentence.lower())  # Lowercase the text and filter out elements other than letters
        sentence = replace_negations(sentence)
        # print("\n-------- [INFO]: REPLACE NEGATIONS --------\n{}".format(sentence))
        tokenized_words = word_tokenize(sentence)
        # print("\n-------- [INFO]: TOKENIZE TO WORDS --------\n{}".format(tokenized_words))
        tagged_sentence = pos_tag(tokenized_words)
        # print("\n-------- [INFO]: POS TAGGING --------\n{}".format(tagged_sentence))
        tagged_sentence_without_stopwords = [w for w in tagged_sentence if not w[0] in stop_words]
        # print("\n-------- [INFO]: REMOVE STOPWORDS --------\n{}".format(tagged_sentence_without_stopwords))

        for word, tag in tagged_sentence_without_stopwords:
            wn_tag = penn_to_wn(tag)

            # print("\n-------- [INFO]: STEMMING RESULT -------- \n{}".format(stemmer.stem(word)))
            # lemma = lemmatizer.lemmatize(stemmer.stem(word), pos=wn_tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                # print("******************** THERE IS NO SYNSETS for lemma: ( {} ) ******************\n".format(lemma))
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            word_sent = swn_synset.pos_score() - swn_synset.neg_score()
            # print("\n###########-------- [INFO]: LEMMA AND SENTIMENT SCORE OF LEMMA --------###########\nLemma: {}, Sent Score: {}".format(lemma, word_sent))

            if word_sent != 0:
                sentiment_sum += word_sent
                tokens_count += 1

    return normalize_sentiment_score(sentiment_sum, tokens_count)


def assign_sentiments(data):
    start = process_time()
    print("\n\n[INFO] ####----- PROCESS STARTED -----####\n\n")
    data["sentiment"] = data["reviewText"].apply(sentiment_analyzer)
    end = process_time()
    print("[INFO] Elapsed time for creating the sentiment column in seconds: {}".format(end - start))
    print("\n\n[INFO] ####----- PROCESS ENDED -----####\n\n")
    return data


def accuracy_calculator(normalized_ratings, sentiment_labels):
    acc = accuracy_score(normalized_ratings, sentiment_labels)
    cnf_matrix = confusion_matrix(normalized_ratings,sentiment_labels)
    clf_report = classification_report(normalized_ratings, sentiment_labels)
    print("\n[INFO] ######----- Report of sentiment analysis w.r.t their ratings -----#####")
    print("\n[INFO] Accuracy: {}\n".format(acc))
    print("\n[INFO] Confusion Matrix:\n{}\n".format(cnf_matrix))
    print("\n[INFO] Classification Report:\n{}\n".format(clf_report))


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# pd.set_option('display.expand_frame_repr', False)
path = "data/All_Beauty.json"
# path = "data/test_data.json"

# data = get_df(path)
# data = data.drop(
#     columns=["reviewTime", "verified", "summary", "vote", "style", "reviewerName", "unixReviewTime", "image", "style"])
# data = data[["overall", "reviewText", "reviewerID", "asin"]].dropna(how="any", axis=0)
# print("Null data: {}".format(data["reviewText"].isnull().any().sum()))
# print(data.tail())
# data["normalizedRatings"] = data["overall"].apply(normalize_ratings)
# # data_with_sentiments = assign_sentiments(data)
# data_with_sentiments = data_with_sentiments.drop(columns=["reviewText", "overall"])
# print("\n\n{}".format(data_with_sentiments))

# data_with_sentiments.to_csv("data/test_data_sentiment.csv", encoding="utf-8", index=False)
# data_with_sentiments.to_csv("data/output/new.csv", encoding="utf-8", index=False)

# data_with_sentiments.to_csv("data/All_Beauty_sentiment.csv", encoding="utf-8", index=False)

data_with_sentiment = pd.read_csv("data/output/All_Beauty_sentiment_V2.csv")
# data_with_sentiment = data_with_sentiment.drop()
print(data_with_sentiment.head())
accuracy_calculator(data_with_sentiment["normalizedRatings"], data_with_sentiment["sentiment"])

# [INFO] ####----- PROCESS STARTED -----####
#
#
# [INFO] Elapsed time for creating the sentiment column in seconds: 1623.488965441
#
#
# [INFO] ####----- PROCESS ENDED -----####
#
