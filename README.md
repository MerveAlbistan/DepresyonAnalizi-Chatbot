# Depression Detection Chatbot

🇬🇧 [English](#english) | 🇹🇷 [Türkçe](#türkçe)

---

<a name="english"></a>
## 🇬🇧 English

### Overview

A chatbot that screens for depressive symptoms based on free-text answers to interview-style questions, built and evaluated on the **DAIC-WOZ** dataset. The project compares a classical TF-IDF + Logistic Regression baseline against fine-tuned BERT models, and ships a working Streamlit demo.

This was built as a learning / portfolio project after graduating in Computer Engineering — the goal was not just a working model, but a documented, reasoned comparison of approaches on a small, real-world clinical dataset.

### Dataset

This project uses the **DAIC-WOZ Depression Database** (part of the AVEC 2017 Depression Sub-Challenge), which contains 189 clinical interview sessions (audio, video features, and transcripts) along with PHQ-8 depression scores.

**The raw data is not included in this repository** due to the dataset's End User License Agreement, which restricts redistribution to individuals who have signed their own agreement with USC.

To obtain access: http://dcapswoz.ict.usc.edu/ (request via the official DAIC-WOZ access form).

If you use this dataset, please cite:

> Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. *The Distress Analysis Interview Corpus of human and computer interviews.* LREC 2014, pp. 3123-3128.
>
> DeVault, D., Artstein, R., Benn, G., Dey, T., Fast, E., Gainer, A., Georgila, K., Gratch, J., Hartholt, A., Lhommet, M., Lucas, G., Marsella, S., Morbini, F., Nazarian, A., Scherer, S., Stratou, G., Suri, A., Traum, D., Wood, R., Xu, Y., Rizzo, A., and Morency, L.-P. (2014). *SimSensei kiosk: A virtual human interviewer for healthcare decision support.* AAMAS '14, Paris.

### Method

1. **Data preparation** — transcripts were filtered to participant-only speech and merged with PHQ8_Binary labels from the official train/dev splits.
2. **Baseline model** — TF-IDF vectorization + Logistic Regression. Class imbalance (77 vs. 30 in training) was handled with `class_weight='balanced'`.
3. **Transformer comparison** — `bert-base-uncased` was fine-tuned in three configurations: full fine-tuning, class-weighted loss, and a frozen-backbone / classifier-only variant combined with a head+tail truncation strategy (first + last tokens, to capture both opening small talk and closing PHQ-related questions).
4. **Demo application** — a Streamlit chat interface asks a sequence of interview-style questions and classifies the combined response with the saved baseline model.

### Results (on dev set, n=34)

| Model | Accuracy | Recall (class 1) | F1 (class 1) |
|---|---|---|---|
| TF-IDF + LR (no class weighting) | 0.68 | 0.00 | 0.00 |
| **TF-IDF + LR (`class_weight='balanced'`)** | **0.56** | **0.45** | **0.40** |
| TF-IDF + LR + bigrams/min_df tuning | 0.59 | 0.36 | 0.36 |
| SVM (linear kernel, balanced) | 0.47 | 0.45 | 0.36 |
| BERT (full fine-tune) | 0.68 | 0.00 | 0.00 |
| BERT (class-weighted loss) | 0.68 | 0.00 | 0.00 |
| BERT (frozen backbone + head/tail truncation) | 0.35 | 1.00 | 0.50* |

*\*Degenerate result — the model predicted the positive class for nearly all samples; not a genuine improvement.*

**Key finding:** no BERT configuration outperformed the simple TF-IDF + Logistic Regression baseline on this dataset. With only 107 training examples, a ~110M-parameter model did not reliably converge to a useful decision boundary, even with a frozen backbone and an adjusted learning rate. This illustrates that classical methods can remain competitive — or superior — in low-resource NLP settings, and that "bigger model" is not automatically "better model."

### Limitations

- **Small dataset.** 107 training / 34 dev examples is small for any supervised method, and especially for transformer fine-tuning.
- **Truncation.** BERT's 512-token limit required cutting most transcripts (avg. length ~6,900 characters); a head+tail strategy was tried but did not outperform the baseline.
- **Domain mismatch in the demo.** The model was trained on long clinical interview transcripts; short chatbot answers produce low-confidence predictions (probabilities close to 0.5).
- **Item-level PHQ-8 scoring** (predicting each of the 8 PHQ-8 sub-scores individually from short text) was considered but not implemented — the training data is too small to support 8 additional models reliably. Flagged here as potential future work.

### ⚠️ Disclaimer

This is a learning/portfolio project, **not a medical or diagnostic tool**. It must not be used to assess real mental health concerns. If you or someone you know is struggling, please consult a qualified professional.

### Setup & Usage

```bash
pip install -r requirements.txt
```

1. Request DAIC-WOZ access and place the split CSVs and per-participant transcripts as described in the data preparation script.
2. Run `data.py` to build `train_data.csv` / `dev_data.csv` from the raw transcripts.
3. Run `LR_train.py` to train and save the baseline model (`model.pkl`, `vectorizer.pkl`).
4. (Optional) Run `LR_test.py` to reproduce the evaluation metrics.
5. (Optional) Run `bert_train.py` to reproduce the BERT experiments.
6. Launch the demo:
   ```bash
   streamlit run app.py
   ```

### Repository structure

```
app.py            # Streamlit chatbot demo
data.py           # transcript extraction & label merging
LR_train.py       # baseline model training (TF-IDF + Logistic Regression)
LR_test.py        # baseline model evaluation
bert_train.py     # BERT fine-tuning experiments
requirements.txt
```

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

### Genel Bakış

**DAIC-WOZ** veri seti üzerinde eğitilmiş, görüşme tarzı sorulara verilen serbest metin cevaplarından depresif belirtileri taramaya çalışan bir chatbot. Proje, klasik bir TF-IDF + Logistic Regression baseline'ını fine-tune edilmiş BERT modelleriyle karşılaştırıyor ve çalışan bir Streamlit demosu sunuyor.

Bu proje, Bilgisayar Mühendisliği mezuniyeti sonrası bir öğrenme/portföy projesi olarak geliştirildi — amaç sadece çalışan bir model değil, küçük ve gerçek bir klinik veri seti üzerinde yöntemlerin belgelenmiş, gerekçelendirilmiş bir karşılaştırmasıydı.

### Veri Seti

Bu proje, PHQ-8 depresyon skorlarıyla birlikte 189 klinik görüşme oturumu (ses, görüntü özellikleri ve transkriptler) içeren **DAIC-WOZ Depresyon Veritabanı**'nı (AVEC 2017 Depresyon Alt Yarışması'nın bir parçası) kullanmaktadır.

**Ham veri, bu repoda yer almamaktadır** — veri setinin Son Kullanıcı Lisans Sözleşmesi, erişimi USC ile kendi sözleşmesini imzalamış kişilerle sınırlandırmaktadır.

Erişim için: http://dcapswoz.ict.usc.edu/ (resmi DAIC-WOZ erişim formu üzerinden).

Bu veri setini kullanırsanız, lütfen şu kaynakları belirtin:

> Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. *The Distress Analysis Interview Corpus of human and computer interviews.* LREC 2014, s. 3123-3128.
>
> DeVault, D., Artstein, R., Benn, G., Dey, T., Fast, E., Gainer, A., Georgila, K., Gratch, J., Hartholt, A., Lhommet, M., Lucas, G., Marsella, S., Morbini, F., Nazarian, A., Scherer, S., Stratou, G., Suri, A., Traum, D., Wood, R., Xu, Y., Rizzo, A., ve Morency, L.-P. (2014). *SimSensei kiosk: Sağlık hizmetleri karar destek sistemi için sanal bir insan görüşmecisi.* AAMAS '14, Paris.

### Yöntem

1. **Veri hazırlama** — transkriptler sadece katılımcı konuşmalarına filtrelendi, resmi train/dev split'lerindeki PHQ8_Binary etiketleriyle birleştirildi.
2. **Baseline model** — TF-IDF vektörleştirme + Logistic Regression. Sınıf dengesizliği (eğitimde 77'ye 30) `class_weight='balanced'` ile ele alındı.
3. **Transformer karşılaştırması** — `bert-base-uncased`, üç farklı konfigürasyonda fine-tune edildi: tam fine-tuning, sınıf ağırlıklı loss, ve dondurulmuş gövde / sadece sınıflandırıcı katmanı + head+tail kırpma stratejisi (görüşmenin hem başındaki hem sonundaki token'ları alarak, hem yüzeysel soruları hem PHQ ile ilgili soruları yakalamak için).
4. **Demo uygulaması** — bir Streamlit sohbet arayüzü sırayla görüşme tarzı sorular sorar, birleştirilmiş cevabı kaydedilen baseline model ile sınıflandırır.

### Sonuçlar (dev set üzerinde, n=34)

| Model | Accuracy | Recall (sınıf 1) | F1 (sınıf 1) |
|---|---|---|---|
| TF-IDF + LR (class weight yok) | 0.68 | 0.00 | 0.00 |
| **TF-IDF + LR (`class_weight='balanced'`)** | **0.56** | **0.45** | **0.40** |
| TF-IDF + LR + bigram/min_df ayarı | 0.59 | 0.36 | 0.36 |
| SVM (doğrusal kernel, balanced) | 0.47 | 0.45 | 0.36 |
| BERT (tam fine-tune) | 0.68 | 0.00 | 0.00 |
| BERT (class-weighted loss) | 0.68 | 0.00 | 0.00 |
| BERT (dondurulmuş gövde + head/tail kırpma) | 0.35 | 1.00 | 0.50* |

*\*Yozlaşmış (degenerate) sonuç — model neredeyse tüm örnekleri pozitif sınıf olarak tahmin etti; gerçek bir iyileşme değil.*

**Temel bulgu:** Bu veri setinde hiçbir BERT konfigürasyonu, basit TF-IDF + Logistic Regression baseline'ını geçemedi. Sadece 107 eğitim örneğiyle, ~110 milyon parametreli bir model, dondurulmuş gövde ve ayarlanmış öğrenme oranına rağmen anlamlı bir karar sınırına güvenilir şekilde yakınsamadı. Bu, düşük kaynaklı NLP senaryolarında klasik yöntemlerin rekabetçi — hatta üstün — kalabileceğini, "daha büyük model" ifadesinin otomatik olarak "daha iyi model" anlamına gelmediğini gösteriyor.

### Sınırlamalar

- **Küçük veri seti.** 107 eğitim / 34 dev örneği, herhangi bir denetimli yöntem için, özellikle transformer fine-tuning için küçük.
- **Kırpma (truncation).** BERT'in 512 token sınırı, çoğu transkripti kesmeyi gerektirdi (ortalama uzunluk ~6.900 karakter); head+tail stratejisi denendi ama baseline'ı geçemedi.
- **Demo'da alan uyuşmazlığı (domain mismatch).** Model, uzun klinik görüşme transkriptleri üzerinde eğitildi; kısa chatbot cevapları düşük güvenilirlikli tahminler (0.5'e yakın olasılıklar) üretiyor.
- **Madde bazlı PHQ-8 skorlama** (8 PHQ-8 alt skorunun her birini kısa metinden ayrı ayrı tahmin etme) değerlendirildi ama uygulanmadı — eğitim verisi, 8 ek modeli güvenilir şekilde desteklemek için çok küçük. Gelecekte yapılabilecekler arasında not edildi.

### ⚠️ Uyarı

Bu bir öğrenme/portföy projesidir, **tıbbi veya teşhis amaçlı bir araç değildir**. Gerçek ruh sağlığı endişelerini değerlendirmek için kullanılmamalıdır. Siz veya tanıdığınız biri zorlanıyorsa, lütfen bir uzmana danışın.

### Kurulum ve Kullanım

```bash
pip install -r requirements.txt
```

1. DAIC-WOZ erişimi talep edin, split CSV'lerini ve katılımcı bazlı transkriptleri veri hazırlama script'inde belirtildiği şekilde yerleştirin.
2. Ham transkriptlerden `train_data.csv` / `dev_data.csv` oluşturmak için `data.py`'yi çalıştırın.
3. Baseline modeli eğitip kaydetmek için `LR_train.py`'yi çalıştırın (`model.pkl`, `vectorizer.pkl`).
4. (Opsiyonel) Değerlendirme metriklerini tekrar üretmek için `LR_test.py`'yi çalıştırın.
5. (Opsiyonel) BERT deneylerini tekrar üretmek için `bert_train.py`'yi çalıştırın.
6. Demoyu başlatın:
   ```bash
   streamlit run app.py
   ```

### Repo yapısı

```
app.py            # Streamlit chatbot demosu
data.py           # transkript çıkarma & etiket birleştirme
LR_train.py       # baseline model eğitimi (TF-IDF + Logistic Regression)
LR_test.py        # baseline model değerlendirmesi
bert_train.py     # BERT fine-tuning deneyleri
requirements.txt
```
