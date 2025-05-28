import pandas as pd
import ast
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# โหลดข้อมูล
df = pd.read_csv("data/tmdb_5000_movies.csv")
df['genres'] = df['genres'].apply(ast.literal_eval)
df['genre_list'] = df['genres'].apply(lambda x: [d['name'] for d in x])
df = df[df['overview'].notnull()]

# สร้าง label
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genre_list'])

# แปลงข้อความด้วย TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(df['overview'])

# แบ่งข้อมูล train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดล
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# บันทึกโมเดล
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("✅ Training และบันทึกโมเดลเสร็จเรียบร้อย")