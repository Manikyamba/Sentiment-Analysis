from model import get_sentiment_recommendations
from flask import Flask , render_template , request
import pickle
import pandas as pd

app = Flask(__name__)

# open a file, where you stored the pickled data
file1 = open(r'pickle\best_recommendation_model.pkl', 'rb')
# dump information to that file
recommender = pickle.load(file1)

file2 = open(r'pickle\best_sentiment_model.pkl', 'rb')
# dump information to that file
logreg_model = pickle.load(file2)

clean_df = pd.read_csv('pickle\clean_df.csv')

file4 = open(r'pickle\tfidf_vectorizer.pkl', 'rb')
# dump information to that file
tfidf_vectorizer = pickle.load(file4)


@app.route("/" , methods =['POST','GET'])
def main():
    if request.method == 'POST':
        return render_template("index.html", placeholder_text="Hello from Post Method")
    if request.method == 'GET':
        return render_template("index.html", placeholder_text="Hello from Get Method")

@app.route("/submit" , methods =['POST'])
def submit():
    username = request.form['username']
    result = get_sentiment_recommendations(username,recommender,clean_df,tfidf_vectorizer,logreg_model)
    return result.to_html()

if __name__ == '__main__':
    app.run()
