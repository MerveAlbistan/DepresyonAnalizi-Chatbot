from datasets import Dataset
import pandas as pd
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = pd.read_csv("train_data.csv")
dev_data = pd.read_csv("dev_data.csv")

train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
dev_dataset = Dataset.from_pandas(dev_data[['text', 'label']])

def tokenize_et(ornek):
    tum_tokenlar = tokenizer(ornek['text'], add_special_tokens=False, truncation=False)['input_ids']
    
    if len(tum_tokenlar) > 510:
        bas = tum_tokenlar[:255]
        son = tum_tokenlar[-255:]
        secilen_tokenlar = bas + son
    else:
        secilen_tokenlar = tum_tokenlar

    input_ids = [101] + secilen_tokenlar + [102]
    
    padding_uzunlugu = 512 - len(input_ids)
    attention_mask = [1] * len(input_ids) + [0] * padding_uzunlugu
    input_ids = input_ids + [0] * padding_uzunlugu
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': [0] * 512
    }
train_dataset = train_dataset.map(tokenize_et)
dev_dataset = dev_dataset.map(tokenize_et)

print(train_dataset[0])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

for isim, parametre in model.named_parameters():
    if not isim.startswith("classifier"):
        parametre.requires_grad = False

training_args = TrainingArguments(
    output_dir="./bert_sonuc",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    learning_rate=1e-3
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    tahminler = np.argmax(logits, axis=1)
    rapor = classification_report(labels, tahminler, output_dict=True)
    return {
        "accuracy": rapor["accuracy"],
        "recall_class1": rapor["1"]["recall"],
        "f1_class1": rapor["1"]["f1-score"]
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        agirliklar = torch.tensor([1.0, 77/30])  # sınıf 0 : sınıf 1 oranı
        loss_fn = nn.CrossEntropyLoss(weight=agirliklar)
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
