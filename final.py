# -*- coding: utf-8 -*-

############################################
# 0. 환경 설정 / 라이브러리 설치
############################################

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#!pip install transformers datasets accelerate scikit-learn pandas sentencepiece
#!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
#!pip install konlpy ipyplot


import json
import random
import re
from urllib import parse, request

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertModel,
)
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
fm.fontManager.addfont(fontpath)
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

from kobert_tokenizer import KoBERTTokenizer
from konlpy.tag import Okt
import ipyplot

############################################
# 1. 데이터 전처리
############################################

data1 = pd.read_excel('감성대화말뭉치(최종데이터)_Training.xlsx', engine='openpyxl')  # 감성말뭉치
data2 = pd.read_excel('감성대화말뭉치(최종데이터)_Validation.xlsx', engine='openpyxl')
corpus = pd.concat([data1,data2], axis=0, ignore_index=True)

corpus.drop(['Unnamed: 0','연령','성별','신체질환','시스템문장1','시스템문장2', '시스템문장3'], axis=1, inplace=True)

corpus.loc[corpus["감정_소분류"] == '만족스러운',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '편안한',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '신뢰하는',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '안도',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '기쁨',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '감사하는',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '신이 난',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '자신하는',"감정_대분류"]='행복'
corpus.loc[corpus["감정_소분류"] == '느긋',"감정_대분류"]='행복'

corpus.loc[corpus["감정_소분류"] == '고립된',"감정_대분류"]='공포'
corpus.loc[corpus["감정_소분류"] == '혼란스러운',"감정_대분류"]='공포'
corpus.loc[corpus["감정_소분류"] == '두려운',"감정_대분류"]='공포'
corpus.loc[corpus["감정_소분류"] == '불안',"감정_대분류"]='공포'
corpus.loc[corpus["감정_소분류"] == '초조한',"감정_대분류"]='공포'
corpus.loc[corpus["감정_소분류"] == '당혹스러운',"감정_대분류"]='공포'

corpus.loc[corpus["감정_소분류"] == '좌절한',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '눈물이 나는',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '우울한',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '슬픔',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '상처',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '실망한',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '후회되는',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '비통한',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '낙담한',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '버려진',"감정_대분류"]='슬픔'
corpus.loc[corpus["감정_소분류"] == '희생된',"감정_대분류"]='슬픔'

index1 = corpus[corpus["감정_소분류"] == '걱정스러운'].index
index2 = corpus[corpus["감정_소분류"] == '스트레스 받는'].index
index3 = corpus[corpus["감정_소분류"] == '노여워하는'].index
index4 = corpus[corpus["감정_소분류"] == '툴툴대는'].index
index5 = corpus[corpus["감정_소분류"] == '성가신'].index
index6 = corpus[corpus["감정_소분류"] == '한심한'].index
index7 = corpus[corpus["감정_소분류"] == '충격 받은'].index
index8 = corpus[corpus["감정_소분류"] == '안달하는'].index
index9 = corpus[corpus["감정_소분류"] == '배신당한'].index
index10 = corpus[corpus["감정_소분류"] == '질투하는'].index
index11 = corpus[corpus["감정_소분류"] == '괴로워하는'].index
index12 = corpus[corpus["감정_소분류"] == '외로운'].index
index13 = corpus[corpus["감정_소분류"] == '조심스러운'].index
index14 = corpus[corpus["감정_소분류"] == '회의적인'].index
index15 = corpus[corpus["감정_소분류"] == '취약한'].index
index16 = corpus[corpus["감정_소분류"] == '마비된'].index
index17 = corpus[corpus["감정_소분류"] == '염세적인'].index
index18 = corpus[corpus["감정_소분류"] == '열등감'].index
index19 = corpus[corpus["감정_소분류"] == '방어적인'].index
index20 = corpus[corpus["감정_소분류"] == '흥분'].index
index21 = corpus.loc[corpus["감정_소분류"] == '억울한'].index
index22 = corpus[corpus["감정_소분류"] == '죄책감의'].index
index23 = corpus.loc[corpus["감정_소분류"] == '가난한, 불우한'].index
index24 = corpus[corpus["감정_소분류"] == '부끄러운'].index
index25 = corpus.loc[corpus["감정_소분류"] == '당황'].index
index26 = corpus.loc[corpus["감정_소분류"] == '남의 시선을 의식하는'].index

corpus.drop(index1, inplace=True)
corpus.drop(index2, inplace=True)
corpus.drop(index3, inplace=True)
corpus.drop(index4, inplace=True)
corpus.drop(index5, inplace=True)
corpus.drop(index6, inplace=True)
corpus.drop(index7, inplace=True)
corpus.drop(index8, inplace=True)
corpus.drop(index9, inplace=True)
corpus.drop(index10, inplace=True)
corpus.drop(index11, inplace=True)
corpus.drop(index12, inplace=True)
corpus.drop(index13, inplace=True)
corpus.drop(index14, inplace=True)
corpus.drop(index15, inplace=True)
corpus.drop(index16, inplace=True)
corpus.drop(index17, inplace=True)
corpus.drop(index18, inplace=True)
corpus.drop(index19, inplace=True)
corpus.drop(index20, inplace=True)
corpus.drop(index21, inplace=True)
corpus.drop(index22, inplace=True)
corpus.drop(index23, inplace=True)
corpus.drop(index24, inplace=True)
corpus.drop(index25, inplace=True)
corpus.drop(index26, inplace=True)

corpus.drop(['상황키워드','감정_소분류','사람문장2', '사람문장3'], axis=1, inplace=True)
corpus.rename(columns={'감정_대분류':'Emotion'}, inplace=True)
corpus.rename(columns={'사람문장1':'Sentence'}, inplace=True)
corpus = corpus[['Sentence','Emotion']]
corpus = corpus.reset_index(drop=True)

data3 = pd.read_excel('한국어_연속적_대화_데이터셋.xlsx', engine='openpyxl')  # 연속적 대화 데이터셋
data3["Unnamed: 2"].value_counts()

data3 = data3[['Unnamed: 1','Unnamed: 2']]
data3.drop([0],axis=0,inplace=True)
data3.rename(columns={'Unnamed: 1':'Sentence','Unnamed: 2':'Emotion'},inplace=True)
data3.replace('ㅍ','공포',inplace=True)
data3.replace(['분','분ㄴ'],'분노',inplace=True)
data3.replace(['ㅈ중립','중림','ㄴ중립','줄'],'분노',inplace=True)
data3 = data3.dropna(how='any')

data4 = pd.read_csv('혐오 데이터.csv', encoding='cp949')  # 혐오 데이터셋
data4 = data4[data4["혐오"] == 1]
data4 = data4[['문장','혐오']]
data4.rename(columns={'문장':'Sentence','혐오':'Emotion'},inplace=True)
data4['Emotion'].loc[data4["Emotion"] == 1]='혐오'

data = pd.concat([corpus, data3, data4], ignore_index=True)
data.to_csv('data.csv', index=False)

############################################
# 2. 전처리 된 데이터 불러오기
############################################

df = pd.read_csv('data.csv')

df = df.rename(columns={
    "Sentence": "sentence",
    "Emotion": "emotion"
})

# 감정 라벨 매핑
target_classes = {
    '공포': 0,
    '놀람': 1,
    '분노': 2,
    '슬픔': 3,
    '중립': 4,
    '행복': 5,
    '혐오': 6
}
id2label = {v: k for k, v in target_classes.items()}
num_labels = len(target_classes)

if df['emotion'].dtype == object:
    df['emotion'] = df['emotion'].map(target_classes).astype(int)

# train / valid split
train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['emotion']
)

print("train label 범위:", train_df['emotion'].min(), train_df['emotion'].max())
print("valid label 범위:", valid_df['emotion'].min(), valid_df['emotion'].max())

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 device:", device)

############################################
# 3. HuggingFace용 Dataset
############################################

class EmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=64):
        self.sentences = sentences.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences.iloc[idx])
        label = int(self.labels.iloc[idx])

        encoded = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


def evaluate_hf(model, dataloader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            preds.extend(pred.cpu().tolist())
            trues.extend(batch['labels'].cpu().tolist())

    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average='macro')
    return acc, macro_f1, np.array(trues), np.array(preds)


def train_one_model(
    model_name,
    train_df,
    valid_df,
    num_labels,
    alias=None,
    epochs=1,
    batch_size=32,
    lr=2e-5,
    max_len=64,
    device_override=None
):
    if alias is None:
        alias = model_name
    print(f"\n=== 모델: {alias} ({model_name}) ===")

    if device_override is not None:
        dev = torch.device(device_override)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 device:", dev)

    assert train_df['emotion'].min() >= 0 and train_df['emotion'].max() < num_labels
    assert valid_df['emotion'].min() >= 0 and valid_df['emotion'].max() < num_labels

    # 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    model.resize_token_embeddings(len(tokenizer))
    model.to(dev)

    print("tokenizer vocab_size:", len(tokenizer))
    print("input embedding size:", model.get_input_embeddings().weight.shape)

    train_dataset = EmotionDataset(
        train_df['sentence'],
        train_df['emotion'],
        tokenizer,
        max_len=max_len
    )
    valid_dataset = EmotionDataset(
        valid_df['sentence'],
        valid_df['emotion'],
        tokenizer,
        max_len=max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in loop:
            batch = {k: v.to(dev) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_acc, val_f1, _, _ = evaluate_hf(model, valid_loader, dev)
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, val_macro_f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1

    print(f"=== {alias} 최종 성능: acc={best_acc:.4f}, macro_f1={best_f1:.4f} ===")
    return best_acc, best_f1, model, tokenizer

############################################
# 4. KoELECTRA 학습 & 기본 예측
############################################

acc_koel, f1_koel, model_koel, tokenizer_koel = train_one_model(
    model_name="monologg/koelectra-base-v3-discriminator",
    alias="KoELECTRA",
    train_df=train_df,
    valid_df=valid_df,
    num_labels=num_labels,
    epochs=1,
    batch_size=32,
    lr=2e-5,
    max_len=64
)

print("KoELECTRA 성능 → acc:", acc_koel, "macro_f1:", f1_koel)

def predict_emotion_hf(sentence, model, tokenizer, id2label, max_len=64):
    model.eval()
    dev = next(model.parameters()).device

    encoded = tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_len
    )
    encoded = {k: v.to(dev) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))

    pred_label = id2label[pred_id]
    return pred_id, pred_label, probs.cpu().numpy()

test_sent = "오늘 저녁은 치킨이닭"
pred_id, pred_label, probs = predict_emotion_hf(
    test_sent,
    model_koel,
    tokenizer_koel,
    id2label
)

print("입력 문장:", test_sent)
print("예측 감정 코드:", pred_id)
print("예측 감정 이름:", pred_label)
print("각 감정별 확률:", probs)

torch.save(model_koel.state_dict(), "./koelectra_emotion_model_state_dict.pt")
print("저장 완료: koelectra_emotion_model_state_dict.pt")

############################################
# 5. KoELECTRA 평가 세부
############################################

def evaluate_with_preds(model, dataloader, device):
    return evaluate_hf(model, dataloader, device)

max_len = 64
batch_size = 32

train_set_koel = EmotionDataset(train_df['sentence'], train_df['emotion'], tokenizer_koel, max_len=max_len)
valid_set_koel = EmotionDataset(valid_df['sentence'], valid_df['emotion'], tokenizer_koel, max_len=max_len)

train_loader_koel = DataLoader(train_set_koel, batch_size=batch_size, shuffle=False)
valid_loader_koel = DataLoader(valid_set_koel, batch_size=batch_size, shuffle=False)

train_acc_koel, train_f1_koel, _, _ = evaluate_with_preds(model_koel, train_loader_koel, device)
valid_acc_koel, valid_f1_koel, y_true_koel, y_pred_koel = evaluate_with_preds(model_koel, valid_loader_koel, device)

results_df = pd.DataFrame([
    ["KoELECTRA-train", train_acc_koel, train_f1_koel],
    ["KoELECTRA-valid", valid_acc_koel, valid_f1_koel],
], columns=["split", "accuracy", "macro_f1"])

print("=== KoELECTRA split별 성능 ===")
print(results_df)

target_names = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]

print("\n=== KoELECTRA 감정별 분류 리포트 ===")
print(classification_report(y_true_koel, y_pred_koel, target_names=target_names))

cm_koel = confusion_matrix(y_true_koel, y_pred_koel)
print("=== KoELECTRA 혼동행렬 (행: 실제, 열: 예측) ===")
print(cm_koel)

plt.figure(figsize=(8, 6))
plt.imshow(cm_koel)
plt.title("KoELECTRA Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(target_names)), labels=target_names, rotation=45)
plt.yticks(ticks=np.arange(len(target_names)), labels=target_names)
plt.colorbar()
plt.tight_layout()
plt.show()

report_dict_koel = classification_report(
    y_true_koel,
    y_pred_koel,
    target_names=target_names,
    output_dict=True
)

f1_scores_koel = [report_dict_koel[label]["f1-score"] for label in target_names]

plt.figure(figsize=(10, 5))
plt.bar(target_names, f1_scores_koel)
plt.title("KoELECTRA Per-Class F1-score")
plt.xlabel("Emotion")
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

############################################
# 6. KoBERT: 데이터 변환 유틸
############################################

class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length,
                 pad=True, pair=False):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair

        self.cls_token = self._tokenizer.cls_token or '[CLS]'
        self.sep_token = self._tokenizer.sep_token or '[SEP]'
        self.pad_token_id = self._tokenizer.pad_token_id

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __call__(self, line):
        text_a = line[0]
        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            assert len(line) == 2
            text_b = line[1]
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[:self._max_seq_length - 2]

        tokens = []
        tokens.append(self.cls_token)
        tokens.extend(tokens_a)
        tokens.append(self.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(self.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        valid_length = len(input_ids)

        if self._pad:
            padding_length = self._max_seq_length - valid_length
            input_ids.extend([self.pad_token_id] * padding_length)
            segment_ids.extend([0] * padding_length)

        return (np.array(input_ids, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np.array(segment_ids, dtype='int32'))


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx,
                 bert_tokenizer, max_len,
                 pad=True, pair=False):

        transform = BERTSentenceTransform(
            bert_tokenizer,
            max_seq_length=max_len,
            pad=pad,
            pair=pair
        )
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i], )

    def __len__(self):
        return len(self.labels)

max_len_kobert = 100
batch_size_kobert = 64
warmup_ratio = 0.1
num_epochs_kobert = 1
max_grad_norm = 1
log_interval = 200
learning_rate_kobert = 5e-5

# KoBERT 토크나이저 & 모델
tokenizer_kobert = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel_kobert = BertModel.from_pretrained('skt/kobert-base-v1')

data_list_train = [
    [q, str(l)] for q, l in zip(train_df['sentence'], train_df['emotion'])
]
data_list_valid = [
    [q, str(l)] for q, l in zip(valid_df['sentence'], valid_df['emotion'])
]

data_train = BERTDataset(data_list_train, 0, 1, tokenizer_kobert, max_len_kobert, pad=True, pair=False)
data_valid = BERTDataset(data_list_valid, 0, 1, tokenizer_kobert, max_len_kobert, pad=True, pair=False)

train_dataloader_kobert = DataLoader(
    data_train,
    batch_size=batch_size_kobert,
    num_workers=0,
    shuffle=True
)
valid_dataloader_kobert = DataLoader(
    data_valid,
    batch_size=batch_size_kobert,
    num_workers=0,
    shuffle=False
)

print("KoBERT Train batch 수:", len(train_dataloader_kobert))
print("KoBERT Valid batch 수:", len(valid_dataloader_kobert))

############################################
# 7. KoBERT Classifier
############################################

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=7,
                 dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.to(token_ids.device),
            return_dict=False
        )
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

device_kobert = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("KoBERT 사용 디바이스:", device_kobert)

kobert_model = BERTClassifier(bertmodel_kobert, dr_rate=0.5).to(device_kobert)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in kobert_model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in kobert_model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }
]

optimizer_kobert = AdamW(optimizer_grouped_parameters, lr=learning_rate_kobert)
loss_fn_kobert = nn.CrossEntropyLoss()

t_total = len(train_dataloader_kobert) * num_epochs_kobert
warmup_step = int(t_total * warmup_ratio)

scheduler_kobert = get_cosine_schedule_with_warmup(
    optimizer_kobert,
    num_warmup_steps=warmup_step,
    num_training_steps=t_total
)

def calc_accuracy(logits, labels):
    max_vals, max_indices = torch.max(logits, 1)
    acc = (max_indices == labels).sum().item() / max_indices.size(0)
    return acc

############################################
# 8. KoBERT 학습 루프
############################################

train_history = []
valid_history = []
loss_history = []

for epoch in range(num_epochs_kobert):
    kobert_model.train()
    train_acc = 0.0
    batch_count = 0
    print(f"\n===== KoBERT Epoch {epoch+1} / {num_epochs_kobert} =====")

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader_kobert)):
        optimizer_kobert.zero_grad()

        token_ids = token_ids.long().to(device_kobert)
        segment_ids = segment_ids.long().to(device_kobert)
        valid_length = valid_length
        label = label.long().to(device_kobert)

        logits = kobert_model(token_ids, valid_length, segment_ids)
        loss = loss_fn_kobert(logits, label)

        loss.backward()
        nn.utils.clip_grad_norm_(kobert_model.parameters(), max_grad_norm)
        optimizer_kobert.step()
        scheduler_kobert.step()

        batch_acc = calc_accuracy(logits, label)
        train_acc += batch_acc
        batch_count += 1

        if batch_id % log_interval == 0:
            print(f"Epoch {epoch+1} Batch {batch_id+1} "
                  f"Loss {loss.item():.4f} Train Acc {train_acc / batch_count:.4f}")

        train_history.append(train_acc / batch_count)
        loss_history.append(loss.item())

    print(f"Epoch {epoch+1} Train Acc: {train_acc / batch_count:.4f}")

    # ----- 검증 -----
    kobert_model.eval()
    valid_acc = 0.0
    valid_batch_count = 0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(valid_dataloader_kobert)):
            token_ids = token_ids.long().to(device_kobert)
            segment_ids = segment_ids.long().to(device_kobert)
            valid_length = valid_length
            label = label.long().to(device_kobert)

            logits = kobert_model(token_ids, valid_length, segment_ids)
            batch_acc = calc_accuracy(logits, label)
            valid_acc += batch_acc
            valid_batch_count += 1

    print(f"Epoch {epoch+1} Valid Acc: {valid_acc / valid_batch_count:.4f}")
    valid_history.append(valid_acc / valid_batch_count)
    
torch.save(kobert_model.state_dict(), "./kobert_emotion_model_state_dict.pt")
print("모델 가중치 저장 완료: kobert_emotion_model_state_dict.pt")

############################################
# 9. KoBERT 평가
############################################

def evaluate_kobert(model, dataloader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for token_ids, valid_length, segment_ids, label in dataloader:
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            vlen = valid_length
            label = label.long().to(device)

            logits = model(token_ids, vlen, segment_ids)
            pred = torch.argmax(logits, dim=-1)

            preds.extend(pred.cpu().tolist())
            trues.extend(label.cpu().tolist())

    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average='macro')
    return acc, macro_f1, np.array(trues), np.array(preds)


kobert_acc, kobert_f1, y_true_kobert, y_pred_kobert = evaluate_kobert(
    kobert_model,
    valid_dataloader_kobert,
    device_kobert
)

print("=== KoBERT Valid 성능 ===")
print("accuracy:", kobert_acc)
print("macro F1:", kobert_f1)

sorted_items = sorted(target_classes.items(), key=lambda x: x[1])
target_names_kobert = [k for k, v in sorted_items]

print("\n=== KoBERT 감정별 분류 리포트 ===")
print(classification_report(
    y_true_kobert,
    y_pred_kobert,
    target_names=target_names_kobert
))

cm_kobert = confusion_matrix(y_true_kobert, y_pred_kobert)
print("=== KoBERT 혼동행렬 (행: 실제, 열: 예측) ===")
print(cm_kobert)

plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))
plt.imshow(cm_kobert)
plt.title("KoBERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(ticks=np.arange(len(target_names_kobert)), labels=target_names_kobert, rotation=45)
plt.yticks(ticks=np.arange(len(target_names_kobert)), labels=target_names_kobert)
plt.colorbar()
plt.tight_layout()
plt.show()

report_dict_kobert = classification_report(
    y_true_kobert,
    y_pred_kobert,
    target_names=target_names_kobert,
    output_dict=True
)

labels_kobert = target_names_kobert
f1_scores_kobert = [report_dict_kobert[label]["f1-score"] for label in labels_kobert]

plt.figure(figsize=(10, 5))
plt.bar(labels_kobert, f1_scores_kobert)
plt.title("KoBERT Per-Class F1-score")
plt.xlabel("Emotion")
plt.ylabel("F1-score")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

############################################
# 10. 두 모델 성능 비교
############################################

compare_df = pd.DataFrame([
    ["KoELECTRA", acc_koel, f1_koel],
    ["KoBERT",    kobert_acc, kobert_f1],
], columns=["model", "accuracy", "macro_f1"])

print("\n=== 모델 성능 비교 ===")
print(compare_df)

############################################
# 11-1. OKT + KoELECTRA 키워드+감정
############################################

def predict_emotion_koelectra(sentence, model, tokenizer, id2label, max_len=64):
    model.eval()
    device = next(model.parameters()).device

    encoded = tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=max_len
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))

    pred_label = id2label[pred_id]
    return pred_id, pred_label, probs.cpu().numpy()


def extract_keywords_with_emotion_koelectra(
    sentence,
    model,
    tokenizer,
    id2label,
    okt,
    max_len=64
):
    # 1) 감정 예측 (KoELECTRA)
    pred_id, pred_label, probs = predict_emotion_koelectra(
        sentence,
        model,
        tokenizer,
        id2label,
        max_len=max_len
    )

    # 2) 형태소 분석
    morphs = okt.pos(sentence, stem=True)

    key_pos = {'Noun', 'Adjective', 'Verb'}
    stopwords = {'하다', '되다', '있다', '그냥', '정말', '진짜'}

    keywords = []
    for word, pos in morphs:
        if pos in key_pos and len(word) > 1 and word not in stopwords:
            keywords.append(word)

    # 중복 제거
    seen = set()
    uniq_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            uniq_keywords.append(w)

    return {
        "sentence": sentence,
        "emotion_id": pred_id,
        "emotion_label": pred_label,
        "emotion_probs": probs,
        "keywords": uniq_keywords,
        "morphs_raw": morphs
    }
okt = Okt()

sentence = "날이 너무 추워져서 감기걸리기 쉬워."

res_koel = extract_keywords_with_emotion_koelectra(
    sentence,
    model_koel,       # KoELECTRA 학습된 모델
    tokenizer_koel,   # KoELECTRA 토크나이저
    id2label,
    okt
)

############################################
# 12. GIPHY 연동 & GIF 추천 (KoElectra)
############################################

GIPHY_URL = "https://api.giphy.com/v1/gifs/search"
API_KEY = "P9SzTTpADkMNu3XjLkzRyMxVGm3MyKDQ"

emotion_to_en = {
    '공포': 'scared',
    '놀람': 'surprised',
    '분노': 'angry',
    '슬픔': 'sad',
    '중립': 'neutral',
    '행복': 'happy',
    '혐오': 'disgust'
}

keywords_ko = res_koel["keywords"][:]

emo_ko = res_koel["emotion_label"]
keywords_ko.append(emo_ko)

emo_en = emotion_to_en.get(emo_ko, "emotion")
keywords_ko.append(emo_en)

seen_kw = set()
keywords_final = []
for w in keywords_ko:
    if w not in seen_kw:
        seen_kw.add(w)
        keywords_final.append(w)

print("최종 GIPHY 검색 키워드:", keywords_final)

all_urls = []

for kw in keywords_final:
    params = parse.urlencode({
        "q": kw,
        "api_key": API_KEY,
        "limit": "25"
    })

    try:
        with request.urlopen(f"{GIPHY_URL}?{params}") as response:
            data = json.loads(response.read())

        results = data.get("data", [])
        print(f"키워드 '{kw}' 검색 결과:", len(results))

        if not results:
            continue

        # 각 키워드 최대 3개씩 선택
        n = len(results)
        target_n = min(3, n)
        idxs = random.sample(range(n), target_n)

        for i in idxs:
            all_urls.append(results[i]["images"]["downsized"]["url"])

    except Exception as e:
        print(f"키워드 '{kw}' 검색 중 에러:", e)

if not all_urls:
    print("GIF 추천 실패 😢 검색 결과 없음")
else:
    # 전체 중복 제거
    all_urls = list(dict.fromkeys(all_urls))

    # 더 이상 전체 개수 제한 안 걸고, 전부 보여주기
    picked = [re.sub(r"media\d", "i", url) for url in all_urls]

    print(f"[총 {len(all_urls)}개 추천]")
    ipyplot.plot_images(picked, img_width=200, show_url=False)

############################################
# 11-2. OKT + KoBERT 키워드 추출
############################################

def predict_emotion_kobert(sentence, model, tokenizer, id2label, max_len=100):
    device = next(model.parameters()).device
    model.eval()

    transform = BERTSentenceTransform(
        tokenizer,
        max_seq_length=max_len,
        pad=True,
        pair=False
    )

    input_ids, valid_length, segment_ids = transform([sentence])
    valid_length_int = int(valid_length)

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(device)
    segment_ids = torch.from_numpy(segment_ids).unsqueeze(0).to(device)
    valid_length = torch.tensor([valid_length_int], dtype=torch.int64).to(device)

    with torch.no_grad():
        logits = model(input_ids, valid_length, segment_ids)
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).cpu().numpy())
        pred_prob = float(probs[0, pred_id].cpu().numpy())

    pred_label = id2label[pred_id]

    return pred_id, pred_label, pred_prob

def extract_keywords_with_emotion_kobert(
    sentence,
    model,
    tokenizer,
    id2label,
    okt,
    max_len=100
):

    # 1) KoBERT 감정 예측
    pred_id, pred_label, pred_prob = predict_emotion_kobert(
        sentence,
        model,
        tokenizer,
        id2label,
        max_len=max_len
    )

    # 2) OKT 형태소 분석
    morphs = okt.pos(sentence, stem=True)

    key_pos = {'Noun', 'Adjective', 'Verb'}
    stopwords = {'하다', '되다', '있다', '정말', '진짜', '근데'}

    keywords = []
    for word, pos in morphs:
        if pos in key_pos and len(word) > 1 and word not in stopwords:
            keywords.append(word)

    # 중복 제거
    seen = set()
    uniq_keywords = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            uniq_keywords.append(w)

    return {
        "sentence": sentence,
        "emotion_id": pred_id,
        "emotion_label": pred_label,
        "emotion_prob": pred_prob,
        "keywords": uniq_keywords,
        "morphs_raw": morphs
    }

sentence = "요즘 날씨도 춥고 기분도 좀 우울해."

res_kobert = extract_keywords_with_emotion_kobert(
    sentence,
    kobert_model,      # 학습된 KoBERT 분류기
    tokenizer_kobert,  # KoBERT 토크나이저
    id2label,
    okt
)

print(res_kobert)

############################################
# 12. GIPHY 연동 & GIF 추천 (KoBERT)
############################################

GIPHY_URL = "https://api.giphy.com/v1/gifs/search"
API_KEY = "P9SzTTpADkMNu3XjLkzRyMxVGm3MyKDQ"

emotion_to_en = {
    '공포': 'scared',
    '놀람': 'surprised',
    '분노': 'angry',
    '슬픔': 'sad',
    '중립': 'neutral',
    '행복': 'happy',
    '혐오': 'disgust'
}

keywords_ko = res_kobert["keywords"][:]

emo_ko = res_kobert["emotion_label"]
keywords_ko.append(emo_ko)

emo_en = emotion_to_en.get(emo_ko, "emotion")
keywords_ko.append(emo_en)

seen_kw = set()
keywords_final = []
for w in keywords_ko:
    if w not in seen_kw:
        seen_kw.add(w)
        keywords_final.append(w)

print("최종 GIPHY 검색 키워드:", keywords_final)

all_urls = []

for kw in keywords_final:
    params = parse.urlencode({
        "q": kw,
        "api_key": API_KEY,
        "limit": "25"
    })

    try:
        with request.urlopen(f"{GIPHY_URL}?{params}") as response:
            data = json.loads(response.read())

        results = data.get("data", [])
        print(f"키워드 '{kw}' 검색 결과:", len(results))

        if not results:
            continue

        # 각 키워드 최대 3개씩 선택
        n = len(results)
        target_n = min(3, n)
        idxs = random.sample(range(n), target_n)

        for i in idxs:
            all_urls.append(results[i]["images"]["downsized"]["url"])

    except Exception as e:
        print(f"키워드 '{kw}' 검색 중 에러:", e)

if not all_urls:
    print("GIF 추천 실패 😢 검색 결과 없음")
else:
    # 전체 중복 제거
    all_urls = list(dict.fromkeys(all_urls))

    # 더 이상 전체 개수 제한 안 걸고, 전부 보여주기
    picked = [re.sub(r"media\d", "i", url) for url in all_urls]

    print(f"[총 {len(all_urls)}개 추천]")
    ipyplot.plot_images(picked, img_width=200, show_url=False)