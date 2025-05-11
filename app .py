import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Başlık
st.title("🎓 Öğrenci Başarı Tahmini Uygulaması")

# Model ve özellikleri yükle
model = joblib.load("student_pass_model.pkl")
features = joblib.load("selected_features.pkl")

# Özellik aralıkları (normalize edilmiş veriye karşılık gelen gerçek aralıklar)
ranges = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "failures": (0, 4),
    "freetime": (1, 5),
    "goout": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20)
}

# Açıklayıcı Türkçe etiketler
labels = {
    "age": "Yaş",
    "Medu": "Anne Eğitim Seviyesi (0-4)",
    "Fedu": "Baba Eğitim Seviyesi (0-4)",
    "failures": "Kaldığı Ders Sayısı",
    "freetime": "Boş Zaman Miktarı (1-5)",
    "goout": "Arkadaşlarla Dışarı Çıkma (1-5)",
    "absences": "Devamsızlık Sayısı",
    "G1": "1. Sınav Notu (1. Dönem)",
    "G2": "2. Sınav Notu (2. Dönem)",
    "G3": "Final Sınavı Notu"
}

# Kullanıcıdan giriş al
input_values = []
st.markdown("### Öğrenci Bilgilerini Giriniz:")
for feat in features:
    label = labels.get(feat, feat)
    min_val, max_val = ranges.get(feat, (0, 1))
    value = st.slider(label, float(min_val), float(max_val), float((min_val + max_val) / 2))
    input_values.append(value)

# Arka planda normalize et
scaler = MinMaxScaler()
scaler.fit([[ranges[f][0] for f in features], [ranges[f][1] for f in features]])
input_normalized = scaler.transform([input_values])

# Tahmin
if st.button("Tahmin Et"):
    result = model.predict(input_normalized)
    if result[0] == 1:
        st.success("🎉 Öğrenci GEÇTİ")
    else:
        st.error("❌ Öğrenci KALDI")
