import pickle

# โหลดโมเดลและอ็อบเจกต์
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# ทดสอบตัวอย่างเรื่องย่อ
examples = [
    "A space crew travels to a distant planet where they encounter strange creatures.",
    "A woman falls in love with her best friend while navigating the ups and downs of college life.",
    "A group of superheroes must save the world from an alien invasion.",
    "A family moves into a haunted house and experiences terrifying supernatural events.",
    "A documentary about the rise and fall of a famous rock band."
]

# ทำนายและแสดงผล
for overview in examples:
    vec = vectorizer.transform([overview])
    pred = model.predict(vec)
    genres = mlb.inverse_transform(pred)[0]
    print("🎬 Overview:", overview)
    print("✅ Predicted Genres:", ", ".join(genres) if genres else "None")
    print("---")