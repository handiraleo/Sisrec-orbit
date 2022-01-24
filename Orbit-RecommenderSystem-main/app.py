import numpy as np
import pandas as pd
from scipy import sparse
from flask import Flask, render_template, request

ratings = pd.read_csv('dataset.csv')

userRatings = ratings.pivot_table(index=['User_id'],columns=['Nama_wisata'],values='Rating')

userRatings = userRatings.dropna(thresh=10).fillna(0)

corrMatrix = userRatings.corr(method='spearman')

#Fungsi untuk membuat list rekomendasi
def get_similar(Genre,Rating):
    similar_ratings = corrMatrix[Genre]*(Rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings

#Deployment    
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['POST'])
def predict():
    wisata = request.form['wisata']
    rating = int(request.form['rating'])
    rekomendasi = get_similar(wisata, rating)
    rekomendasi = list(rekomendasi[1:6].index)
        
    return render_template('home.html', output=rekomendasi)
        
    
if __name__ == "__main__":
    app.run(debug=True)
