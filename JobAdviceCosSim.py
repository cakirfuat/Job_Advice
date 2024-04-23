import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data_all = pd.read_csv(".venv/Data/allJobs.csv")
data = data_all.head(2000).copy()
data.dropna(subset=['Description'], inplace=True)
user_input = "sorting ml python spss data"
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(data['Description'])
user_input_tfidf = vectorizer.transform([user_input])
cos_sim = cosine_similarity(X_tfidf, user_input_tfidf)
data['cos_sim'] = cos_sim
sorted_data = data.sort_values(by='cos_sim', ascending=False)
top_recommendations = sorted_data.head(5)['Job-Title'].tolist()
for i, job_title in enumerate(top_recommendations, start=1):
    print("Önerilen iş ünvanları:")
    print(f"{i}. {job_title}")



