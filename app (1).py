import streamlit as st
import numpy as np
import joblib

# Başlık
st.title("🎓 Öğrenci Başarı Tahmini Uygulaması")

# Model ve özellikleri yükle
model = joblib.load("student_pass_model.pkl")
features = joblib.load("selected_features.pkl")

# Kullanıcıdan giriş al
input_values = []
st.markdown("### Giriş Bilgilerini Girin:")
for feat in features:
    value = st.slider(f"{feat}", 0.0, 1.0, 0.5)
    input_values.append(value)

# Tahmin
if st.button("Tahmin Et"):
    result = model.predict([input_values])
    if result[0] == 1:
        st.success("🎉 Öğrenci GEÇTİ")
    else:
        st.error("❌ Öğrenci KALDI")
