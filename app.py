
import streamlit as st
import pickle

st.title("🎬 ระบบแนะนำแนวหนังจากเรื่อง")

# โหลด model, vectorizer, และ mlb
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)

# รับ input จากผู้ใช้
overview = st.text_area("📄 กรอกเรื่องย่อหนัง")

if st.button("🔍 ทำนายแนวหนัง"):
    if overview.strip():
        vec = vectorizer.transform([overview])
        pred = model.predict(vec)
        genres = mlb.inverse_transform(pred)[0]
        st.success("✅ แนวหนังที่ระบบทำนายคือ: " + ", ".join(genres))
    else:
        st.warning("⚠️ กรุณาใส่หนังจากเรื่อง")
