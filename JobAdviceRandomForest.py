import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Veri setini yükleme
data_all = pd.read_csv(".venv/Data/allJobs.csv")
data = data_all.head(1000).copy()

# NaN değerleri kaldırma
data.dropna(subset=['Description'], inplace=True)

# Veri setini iş ünvanları (labels) ve ilan açıklamaları (features) olarak ayırma
X = data['Description']
y = data['Job-Title']

# Veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# TF-IDF vektörleştiriciyi tanımlama ve uygulama
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# RandomForestClassifier modeli için parametre aralığını tanımlayın
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# GridSearchCV kullanarak en iyi parametreleri bulun
rf_model = RandomForestClassifier()
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# En iyi parametreleri ve en iyi skorları yazdırın
print("En iyi parametreler:", grid_search.best_params_)
print("En iyi skor:", grid_search.best_score_)

# En iyi modeli alın
best_model = grid_search.best_estimator_

# Test seti üzerinde model performansını değerlendirme
y_pred = best_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Test seti doğruluğu:", accuracy)

# Kullanıcının girdiği metni TF-IDF vektörüne dönüştürün
user_input_tfidf = vectorizer.transform([sample])

# Modeli kullanarak iş ünvanı önerisi yapın
predicted_job_titles = best_model.predict(user_input_tfidf)

print("Önerilen iş ünvanı:", predicted_job_titles)