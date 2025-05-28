# 🎬 เว็ปแอปจำแนกประเภทภาพยนตร์ Movie Genre Classifier 

โปรเจกต์นี้สร้างระบบจำแนกประเภทของภาพยนตร์จากเรื่องย่อ (movie overview) โดยใช้โมเดล Machine Learning (Logistic Regression) ร่วมกับ TF-IDF และ MultiLabelBinarizer ในการจัดการกับข้อมูลหลายประเภท (Multi-label Classification)

## 🎯 จุดประสงค์ของโปรเจกต์

เพื่อพัฒนาระบบที่สามารถวิเคราะห์และจำแนกแนวภาพยนตร์จากข้อความเรื่องย่อโดยอัตโนมัติ ซึ่งเป็นการประยุกต์ใช้เทคนิค Natural Language Processing (NLP) และ Machine Learning เพื่อช่วยในงานแนะนำเนื้อหา การจัดหมวดหมู่ หรือการสร้างระบบแนะนำ (recommendation system) โดยระบบนี้สามารถทำนายได้มากกว่าหนึ่งประเภทต่อเรื่อง (multi-label classification)



## 📦 ข้อมูลที่ใช้ Dataset
- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- ประกอบด้วย:
  - `tmdb_5000_movies.csv` – ข้อมูลเรื่องย่อ, ชื่อเรื่อง, ประเภท
  - `tmdb_5000_credits.csv` – ข้อมูลนักแสดงและทีมงาน

## 🧠 เทคนิคที่ใช้
- Text Preprocessing ด้วย TF-IDF (TfidfVectorizer)
- Multi-label classification ด้วย MultiLabelBinarizer
- โมเดล Logistic Regression แบบ One-vs-Rest
- การ deploy ผ่าน Streamlit ให้ใช้งานบน Web ได้ง่าย

## 🎭 ประเภทของหนังที่สามารถจำแนกได้
ระบบสามารถจำแนกภาพยนตร์ได้ทั้งหมด **20 ประเภท** ดังนี้:

- Action  
- Adventure  
- Animation  
- Comedy  
- Crime  
- Documentary  
- Drama  
- Family  
- Fantasy  
- Foreign  
- History  
- Horror  
- Music  
- Mystery  
- Romance  
- Science Fiction  
- TV Movie  
- Thriller  
- War  
- Western

## ⚙️ วิธีติดตั้งและใช้งาน

### 1. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install -r requirements.txt
```

### 2. รัน Web App ด้วย Streamlit
```bash
streamlit run app.py
```

### 3. กรอกเรื่องย่อหนังในกล่องข้อความ แล้วกดปุ่มเพื่อให้ระบบทำนายแนวหนัง
![image](https://github.com/user-attachments/assets/39ebe4fe-8049-4d6c-b754-2b178b1ee83d)


## 👨‍💻 ผู้พัฒนา

ชื่อ: นายศุภกฤษฏิ์ ทิมกลับ

โครงงานนี้เป็นส่วนหนึ่งของวิชา [AI/ระบบสมองกลฝังตัวและอิเล็กทรอนิกส์สื่อสาร]

ส่งวันที่ 28 พ.ค. 2568




