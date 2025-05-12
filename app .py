import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Öğrenci Geçme Tahmini", layout="centered")

# Başlık
st.title("🎓 Öğrenci Başarı (Geçme) Tahmini Uygulaması")

# Model ve özellikleri yükle
model = joblib.load("student_pass_model.pkl")
features = joblib.load("selected_features.pkl")

# Özellik aralıkları (normalize edilirken kullanılacak)
ranges = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "failures": (0, 4),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Walc": (1, 5),
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20)
}

# Açıklayıcı Türkçe etiketler
labels = {
    "age": "Yaş",
    "Medu": "Anne Eğitim Düzeyi (0: Yok - 4: Üniversite)",
    "Fedu": "Baba Eğitim Düzeyi (0: Yok - 4: Üniversite)",
    "failures": "Daha Önce Kaldığı Ders Sayısı",
    "freetime": "Okul Sonrası Boş Zaman (1: Çok Az - 5: Çok Fazla)",
    "goout": "Arkadaşlarla Dışarı Çıkma (1: Nadiren - 5: Sık Sık)",
    "Walc": "Hafta Sonu Alkol Kullanımı (1: Yok - 5: Çok Fazla)",
    "absences": "Toplam Devamsızlık Günü (0 - 93)",
    "G1": "1. Sınav Notu (1. Dönem, 0-20)",
    "G2": "2. Sınav Notu (2. Dönem, 0-20)",
    "G3": "Final Sınavı Notu (0-20)"
}

# Girişleri al
input_values = []
st.markdown("### 📋 Öğrenci Bilgilerini Giriniz")
for feat in features:
    label = labels.get(feat, feat)
    min_val, max_val = ranges.get(feat, (0, 1))
    value = st.slider(label, float(min_val), float(max_val), float((min_val + max_val) / 2))
    input_values.append(value)

# Normalize et
min_vals = []
max_vals = []

for f in features:
    if f in ranges:
        min_vals.append(ranges[f][0])
        max_vals.append(ranges[f][1])
    else:
        min_vals.append(0)
        max_vals.append(1)

scaler = MinMaxScaler()
scaler.fit([min_vals, max_vals])
input_normalized = scaler.transform([input_values])

# Tahmin yap
if st.button("📊 Tahmin Et"):
    result = model.predict(input_normalized)
    if result[0] == 1:
        st.success("🎉 Tahmin: Öğrenci **GEÇTİ** 🎓")
    else:
        st.error("❌ Tahmin: Öğrenci **KALDI**")

# Devamsızlıktan otomatik kalma kontrolü (örnek eşik: 20 gün)
absences_index = features.index("absences") if "absences" in features else None
absences_value = input_values[absences_index] if absences_index is not None else 0

# Tahmin yap
if st.button("📊 Tahmin Et"):
    if absences_value > 20:
        st.warning("📛 Öğrenci devamsızlıktan KALDI (20 günden fazla)")
    else:
        result = model.predict(input_normalized)
        if result[0] == 1:
            st.success("🎉 Tahmin: Öğrenci **GEÇTİ** 🎓")
        else:
            st.error("❌ Tahmin: Öğrenci **KALDI**")

