import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("NSTD_Assignment.csv")

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'female':0, 'male':1}
    return word_dict[word]

df['Gender'] = df['Gender'].apply(lambda x : convert_to_int(x))

X = df[["Gender", "Age"]]
y = df["Risk_Profile"]

clf = LinearRegression() 
clf.fit(X, y)

import joblib

joblib.dump(clf, "clf1.pkl")


