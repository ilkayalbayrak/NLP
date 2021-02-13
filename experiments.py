import pandas as pd
from tools import accuracy_calculator

# Load the prepared CSV file
data_with_sentiment = pd.read_csv("data/output/All_Beauty_sentiment_V2.csv")
# Calculate if the overall rating of the product aligns with its sentiment score
accuracy_calculator(data_with_sentiment["normalizedRatings"], data_with_sentiment["sentiment"])

# [INFO] ######----- Report of sentiment analysis w.r.t their ratings -----#####
#
# [INFO] Accuracy: 0.639877502385792
#
#
# [INFO] Confusion Matrix:
# [[ 23643   8583  27284]
#  [  7917   3512  18111]
#  [ 37466  34225 210205]]
#
#
# [INFO] Classification Report:
#               precision    recall  f1-score   support
#
#           -1       0.34      0.40      0.37     59510
#            0       0.08      0.12      0.09     29540
#            1       0.82      0.75      0.78    281896
#
#     accuracy                           0.64    370946
#    macro avg       0.41      0.42      0.41    370946
# weighted avg       0.69      0.64      0.66    370946


# Drop the rows that zero rating or zero sentiment label.
index_names = data_with_sentiment[(data_with_sentiment["normalizedRatings"] == 0) | (data_with_sentiment["sentiment"] == 0)].index
print("\n[INFO] #####----- Scores; when neutral ratings and sentiments are excluded -----#####")
print("\n[INFO] There are {} rows with either score 0(neutral) rating or score 0 sentiment.".format(len(index_names)))
data_with_sentiment.drop(index_names, inplace=True)
accuracy_calculator(data_with_sentiment["normalizedRatings"], data_with_sentiment["sentiment"])

# [INFO] #####----- Scores; when neutral ratings and sentiments are excluded -----#####
#
# [INFO] There are 72348 rows with either score 0(neutral) rating or score 0 sentiment.
#
# [INFO] ######----- Report of sentiment analysis w.r.t their ratings -----#####
#
# [INFO] Accuracy: 0.7831532696133263
#
#
# [INFO] Confusion Matrix:
# [[ 23643  27284]
#  [ 37466 210205]]
#
#
# [INFO] Classification Report:
#               precision    recall  f1-score   support
#
#           -1       0.39      0.46      0.42     50927
#            1       0.89      0.85      0.87    247671
#
#     accuracy                           0.78    298598
#    macro avg       0.64      0.66      0.64    298598
# weighted avg       0.80      0.78      0.79    298598

