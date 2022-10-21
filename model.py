import pandas as pd
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("NSTD_Assignment.csv")

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'female':0, 'male':1}
    return word_dict[word]

df['Gender'] = df['Gender'].apply(lambda x : convert_to_int(x))

X = df[["Gender", "Age"]]
y = df["Risk_Profile"]

clf = GaussianNB() 
clf.fit(X, y)

import joblib

joblib.dump(clf, "clf1.pkl")
