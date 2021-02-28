# import pandas as pd
from tools import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = "data/All_Beauty.json"
# path = "data/test_data.json"
data = get_df(path)

# ---------------------------- Don't drop the "image" column when using the "test data",-------------------------
# ---------------------it causes an error when there is no "image" column in the test data-------------------------
data = data.drop(columns=["reviewTime", "verified", "summary", "vote", "style", "reviewerName", "unixReviewTime", "image", "style"])
data = data[["overall", "reviewText", "reviewerID", "asin"]].dropna(how="any", axis=0)

# print("Null data: {}".format(data["reviewText"].isnull().any().sum()))
data["normalizedRatings"] = data["overall"].apply(normalize_ratings)
data_with_sentiments = assign_sentiments(data)
data_with_sentiments = data_with_sentiments.drop(columns=["reviewText", "overall"])
data_with_sentiments.to_csv("data/output/test3.csv", encoding="utf-8", index=False)

# [INFO] ####----- PROCESS STARTED -----####
#
# [INFO] Assigning sentiments took 27.05 minutes
#
# [INFO] ####----- PROCESS ENDED -----####


