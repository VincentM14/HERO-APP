from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf1.pkl")
        
        # Get values through input bars
        Gender = request.form.get("Gender")
        Age = request.form.get("Age")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Gender, Age]], columns = ["Gender", "Age"])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)