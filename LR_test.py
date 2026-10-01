import pandas as pd
import joblib
from sklearn.metrics import classification_report

dev_data = pd.read_csv("dev_data.csv")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

metin = dev_data['text']
x_dev = vectorizer.transform(metin)
y_dev = dev_data['label']

"""y_pred = model.predict(x_dev)
print(classification_report(y_dev, y_pred))
"""

y_probs = model.predict_proba(x_dev)
depresif_olasiliklari = y_probs[:, 1]

KARAR_ESIGI = 0.49  

y_pred_custom = (depresif_olasiliklari >= KARAR_ESIGI).astype(int)

print(f"--- KARAR EŞİĞİ ({KARAR_ESIGI}) İLE TEST SONUÇLARI ---")
print(classification_report(y_dev, y_pred_custom))