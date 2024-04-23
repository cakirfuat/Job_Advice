import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)

data_all = pd.read_csv(".venv/Data/allJobs.csv")
data = data_all.head(1000).copy()

data['Description'] = data['Description'].dropna()
data.info()
# Veri setini iş ünvanları (labels) ve ilan açıklamaları (features) olarak ayırma
X = data['Description']
y = data['Job-Title']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF vektörleştiriciyi tanımlama ve uygulama
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Kullanıcıdan kabiliyetlerinizi metin olarak alın
sample = "sorting"

# Sınıflandırma modelini oluşturma ve eğitme
model = LogisticRegression(max_iter=100)
model.fit(X_train_tfidf, y_train)

# Kullanıcının girdiği metni TF-IDF vektörüne dönüştürün
user_input_tfidf = vectorizer.transform([sample])

# Modeli kullanarak iş ünvanı önerisi yapın
predicted_job_title = model.predict(user_input_tfidf)

print("Önerilen iş ünvanı:", predicted_job_title)