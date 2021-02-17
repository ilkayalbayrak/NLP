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
index_names = data_with_sentiment[(data_with_sentiment["normalizedRatings"] == 0)].index
print("\n[INFO] #####----- Scores; when neutral ratings and sentiments are excluded -----#####")
print("\n[INFO] There are {} rows with either score 0(neutral) rating.".format(len(index_names)))
data_with_sentiment.drop(index_names, inplace=True)
accuracy_calculator(data_with_sentiment["normalizedRatings"], data_with_sentiment["sentiment"])

# [INFO] #####----- Scores; when neutral ratings are excluded -----#####
#
# [INFO] There are 29540 rows with either score 0(neutral) rating.
#
# [INFO] ######----- Report of sentiment analysis w.r.t their ratings -----#####
#
# [INFO] Accuracy: 0.6849557418440215
#
#
# [INFO] Confusion Matrix:
# [[ 23643   8583  27284]
#  [     0      0      0]
#  [ 37466  34225 210205]]
#
#
# [INFO] Classification Report:
#               precision    recall  f1-score   support
#
#           -1       0.39      0.40      0.39     59510
#            0       0.00      0.00      0.00         0
#            1       0.89      0.75      0.81    281896
#
#     accuracy                           0.68    341406
#    macro avg       0.42      0.38      0.40    341406
# weighted avg       0.80      0.68      0.74    341406
