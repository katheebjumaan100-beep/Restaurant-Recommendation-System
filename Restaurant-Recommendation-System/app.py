import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("zomato.csv")

# Keep useful columns
df = df[['Name', 'Cuisines', 'Area', 'Dinner Ratings', 'AverageCost']]

# Rename columns
df.columns = ['name', 'cuisines', 'location', 'rating', 'cost']

# Remove missing values
df.dropna(inplace=True)

# Convert everything to string
df['cuisines'] = df['cuisines'].astype(str)
df['location'] = df['location'].astype(str)
df['rating'] = df['rating'].astype(str)

# Create combined feature column
df['features'] = (
    df['cuisines'] + " " +
    df['location'] + " " +
    df['rating']
)

# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(stop_words='english')

feature_vectors = vectorizer.fit_transform(df['features'])

# Similarity matrix
similarity = cosine_similarity(feature_vectors)

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------

def recommend_restaurant(user_input):

    user_input = user_input.lower()

    matches = []

    for i in range(len(df)):

        if user_input in df['cuisines'][i].lower():

            matches.append(i)

    recommendations = []

    for idx in matches[:5]:

        recommendations.append({
            "name": df.iloc[idx]['name'],
            "cuisine": df.iloc[idx]['cuisines'],
            "rating": df.iloc[idx]['rating'],
            "location": df.iloc[idx]['location'],
            "cost": df.iloc[idx]['cost']
        })

    return recommendations

# -----------------------------
# FLASK APP
# -----------------------------

app = Flask(__name__)

@app.route('/')
def home():

    return render_template("index.html")

@app.route('/recommend', methods=['POST'])
def recommend():

    food = request.form['food']

    results = recommend_restaurant(food)

    return render_template(
        "index.html",
        recommendations=results,
        food=food
    )

# -----------------------------
# RUN APP
# -----------------------------

if __name__ == '__main__':

    app.run(debug=True)