import streamlit as st
import joblib
import sys
import sklearn


st.title("Depression Screening Chatbot")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

if 'soru_index' not in st.session_state:
    st.session_state.soru_index = 0

if 'cevaplar' not in st.session_state:
    st.session_state.cevaplar = []


sorular = [
    "How are you doing today?",
    "How have you been feeling lately?",
    "How easy is it for you to get a good night's sleep?",
    "Do you feel down or hopeless often?",
    "What are some things that usually put you in a good mood?",
    "Tell me about something you did recently that you really enjoyed.",
    "How close are you to your family?",
    "what are some things you wish you could change about yourself"
]

for i in range(st.session_state.soru_index):
    st.chat_message("assistant").write(sorular[i])
    st.chat_message("user").write(st.session_state.cevaplar[i])

if st.session_state.soru_index < len(sorular):
    st.chat_message("assistant").write(sorular[st.session_state.soru_index])
    cevap = st.chat_input("Write your answer...")
    
    if cevap:
        st.session_state.cevaplar.append(cevap)
        st.session_state.soru_index += 1
        st.rerun()

else:
    tum_cevaplar = ' '.join(st.session_state.cevaplar)
    
    x_yeni = vectorizer.transform([tum_cevaplar])

    olasiliklar = model.predict_proba(x_yeni)[0]
    depresif_olasiligi = olasiliklar[1]

    KARAR_ESIGI = 0.49  
    
    if depresif_olasiligi >= KARAR_ESIGI:
        tahmin = 1
    else:
        tahmin = 0
        
    st.write("---")

    if tahmin == 1:
        st.warning(f"Depresif belirtiler riski tespit edildi. (Depresif Olasılığı: %{depresif_olasiligi * 100:.1f})")
        st.caption("⚠️ Bu araç yalnızca bir öğrenme/portföy projesidir, tıbbi teşhis amacı taşımaz. Gerçek bir endişeniz varsa lütfen bir uzmana danışın.")
    else:
        st.success(f"Belirgin bir depresif belirti tespit edilmedi. (Depresif Olasılığı: %{depresif_olasiligi * 100:.1f})")
        st.caption("⚠️ Bu araç yalnızca bir öğrenme/portföy projesidir, tıbbi teşhis amacı taşımaz. Gerçek bir endişeniz varsa lütfen bir uzmana danışın.")

