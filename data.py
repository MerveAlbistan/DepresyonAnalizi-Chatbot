import pandas as pd


TRANSKRIPT_KLASORU = "transkriptler"  # DAIC-WOZ transkript dosyalarının bulunduğu klasör

def transkript_oku(dosya_yolu):
    df = pd.read_csv(dosya_yolu, sep='\t')

    participant = df[df['speaker'] == 'Participant']

    sentences = participant['value'].dropna()
    text = ' '.join(sentences)

    return(text)

def split_oku(split_csv_yolu,cikti_adi):
    df = pd.read_csv(split_csv_yolu)
    split_data = df[['Participant_ID', 'PHQ8_Binary']]

    sonuclar = []
    eksik_dosyalar = []

    for index, row in split_data.iterrows():
        id = row['Participant_ID']
        label = row['PHQ8_Binary']
        dosya_yolu = f"{TRANSKRIPT_KLASORU}/{id}_TRANSCRIPT.csv"
        
        try:
            metin = transkript_oku(dosya_yolu)
            sonuclar.append({'Participant_ID': id, 'text': metin, 'label': label})
        except FileNotFoundError:
            eksik_dosyalar.append(id)

    veri = pd.DataFrame(sonuclar)
    veri.to_csv(cikti_adi, index=False)

    print("Eksik dosyalar:", eksik_dosyalar)
    print("Toplanan satır sayısı:", len(veri))

    return veri


train_data = split_oku("train_split_Depression_AVEC2017.csv", "train_data.csv")
dev_data = split_oku("dev_split_Depression_AVEC2017.csv", "dev_data.csv")