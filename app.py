#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 11:57:21 2025

@author: parkseongheon
"""

import os
import random
import json
import re

import torch
from torch import nn

import streamlit as st
import requests

from transformers import BertModel, AutoTokenizer, AutoModelForSequenceClassification
from kobert_tokenizer import KoBERTTokenizer
from konlpy.tag import Okt

import numpy as np
import html

############################################
# 0. 랜덤 시드
############################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

############################################
# 1. BERT 관련 유틸
############################################

class BERTSentenceTransform:
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=False):
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
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
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

############################################
# 2. KoBERT 감정 예측 + 키워드 추출 함수
############################################

target_classes = {
    '공포': 0,
    '놀람': 1,
    '분노': 2,
    '슬픔': 3,
    '중립': 4,
    '행복': 5,
    '혐오': 6
}
idx2label = {v: k for k, v in target_classes.items()}

def predict_emotion_kobert(sentence, model, tokenizer, transform, idx2label, max_len=100):
    device = next(model.parameters()).device
    model.eval()

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

    pred_label = idx2label[pred_id]
    return pred_id, pred_label, pred_prob


def extract_keywords_with_emotion_kobert(sentence, model, tokenizer, transform, idx2label, okt, max_len=100):
    # 1) KoBERT 감정 예측
    pred_id, pred_label, pred_prob = predict_emotion_kobert(
        sentence,
        model,
        tokenizer,
        transform,
        idx2label,
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

############################################
# 2-1. KoELECTRA 감정 예측 + 키워드 추출 함수
############################################

def predict_emotion_koelectra(sentence, model, tokenizer, idx2label, max_len=64):
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

    pred_label = idx2label[pred_id]
    pred_prob = float(probs[pred_id].cpu().numpy())

    return pred_id, pred_label, pred_prob


def extract_keywords_with_emotion_koelectra(
    sentence,
    model,
    tokenizer,
    idx2label,
    okt,
    max_len=64
):
    # 1) 감정 예측 (KoELECTRA)
    pred_id, pred_label, pred_prob = predict_emotion_koelectra(
        sentence,
        model,
        tokenizer,
        idx2label,
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
        "emotion_prob": pred_prob,
        "keywords": uniq_keywords,
        "morphs_raw": morphs
    }


############################################
# 3. GIPHY GIF 추천 함수
############################################

EMOTION_TO_EN = {
    '공포': 'scared',
    '놀람': 'surprised',
    '분노': 'angry',
    '슬픔': 'sad',
    '중립': 'neutral',
    '행복': 'happy',
    '혐오': 'disgust'
}

def recommend_gifs_from_result(res_kobert, api_key, limit_per_kw=3, total_limit=12):
    giphy_url = "https://api.giphy.com/v1/gifs/search"

    if not api_key:
        return [], "GIPHY API Key가 설정되지 않았습니다."

    keywords_ko = res_kobert["keywords"][:]
    emo_ko = res_kobert["emotion_label"]
    keywords_ko.append(emo_ko)
    emo_en = EMOTION_TO_EN.get(emo_ko, "emotion")
    keywords_ko.append(emo_en)

    seen_kw = set()
    keywords_final = []
    for w in keywords_ko:
        if w not in seen_kw:
            seen_kw.add(w)
            keywords_final.append(w)

    all_urls = []

    for kw in keywords_final:
        params = {
            "q": kw,
            "api_key": api_key,
            "limit": "25",
            "rating": "pg-13"
        }
        try:
            resp = requests.get(giphy_url, params=params, timeout=5)
            if resp.status_code != 200:
                continue
            data = resp.json()
            results = data.get("data", [])

            if not results:
                continue

            n = len(results)
            target_n = min(limit_per_kw, n)
            idxs = random.sample(range(n), target_n)

            for i in idxs:
                url = results[i]["images"]["downsized"]["url"]
                all_urls.append(url)

        except Exception as e:
            print(f"GIPHY 검색 중 에러 (kw={kw}):", e)

    if not all_urls:
        return [], "GIF 검색 결과가 없습니다."

    all_urls = list(dict.fromkeys(all_urls))

    if len(all_urls) > total_limit:
        all_urls = random.sample(all_urls, total_limit)

    picked = [re.sub(r"media\d", "i", url) for url in all_urls]

    return picked, f"총 {len(picked)}개 GIF 추천"

############################################
# 4. 모델/토크나이저/OKT 로드 (캐시)
############################################

@st.cache_resource
def load_kobert_model_and_tokenizer(max_len=100, model_path="./kobert_emotion_model_state_dict.pt"):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1')

    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"경고: {model_path} 파일이 존재하지 않습니다.")

    transform = BERTSentenceTransform(
        tokenizer,
        max_seq_length=max_len,
        pad=True,
        pair=False
    )

    okt = Okt()

    return model, tokenizer, transform, okt, device

@st.cache_resource
def load_koelectra_model_and_tokenizer(max_len=64, model_path="./koelectra_emotion_model_state_dict.pt"):
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    model = AutoModelForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        num_labels=len(target_classes)
    ).to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"경고: {model_path} 파일이 존재하지 않습니다.")

    okt = Okt()
    return model, tokenizer, okt, device

############################################
# 5. Streamlit UI
############################################

def inject_chat_css():
    st.markdown("""
    <style>
    .chat-wrapper {
        max-width: 700px;
        margin: 0.5rem 0 1.0rem 0;
    }

    .chat-row {
        display: flex;
        margin-bottom: 0.8rem;
    }

    .chat-bubble {
        padding: 18px 24px;
        border-radius: 26px;
        max-width: 100%;
        font-size: 1.2rem;
        line-height: 1.6;
        word-wrap: break-word;
        word-break: keep-all;
    }

    .chat-bubble.bot {
        background: #f1f0f0;
        color: #111;
        border-bottom-left-radius: 8px;
    }

    .chat-bubble.user {
        margin-left: auto;
        background: linear-gradient(
            to bottom,
            #a855f7,
            #7c3aed
        );
        color: #fff;
        border-bottom-right-radius: 6px;
    }
                            
    .chat-meta {
        font-size: 0.8rem;
        color: #999;
        margin-top: 0.1rem;
        margin-bottom: 0.9rem;
    }

    .keyboard-img {
        opacity: 0.9;
        margin-top: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="감정 분석 + GIF 추천 서비스", layout="wide")
    inject_chat_css()
    
    # ===== 세션 상태 초기화 =====
    if 'sent_once' not in st.session_state:
        st.session_state['sent_once'] = False
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    
    # ===== 사이드바: 설정 =====
    
    backend = st.sidebar.radio(
        "사용할 모델",
        ["KoBERT", "KoELECTRA"],
        index=0
    )
    
    max_len = st.sidebar.slider(
        "최대 토큰 길이",
        min_value=32,
        max_value=256,
        value=100 if backend == "KoBERT" else 64,
        step=4
    )
    
    giphy_api_key = st.sidebar.text_input(
        "GIPHY API Key",
        "P9SzTTpADkMNu3XjLkzRyMxVGm3MyKDQ",
        type="password"
    )
    
    # ===== 모델 로딩 =====
    with st.spinner("모델 로딩 중... (최초 1회)"):
        if backend == "KoBERT":
            model, tokenizer, transform, okt, device = load_kobert_model_and_tokenizer(
                max_len=max_len
            )
        else:
            model, tokenizer, okt, device = load_koelectra_model_and_tokenizer(
                max_len=max_len
            )
            transform = None

    st.session_state['backend'] = backend

    # ===== 메인 영역 =====
    st.markdown("## 📩 감정 분석 + GIF 추천 서비스")

    chat_container = st.container()

    # ===== 입력창 & 버튼 =====
    default_text = "요즘 날씨도 춥고 기분도 좀 우울해."
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        sentence = st.text_input(
            "메시지 입력",
            value=default_text,
            label_visibility="collapsed"
        )
    with col_btn:
        analyze_clicked = st.button("GIF 추천받기")
    
    # 1단계: 버튼 클릭 처리 & 감정 분석
    if analyze_clicked:
        if not sentence.strip():
            st.warning("문장을 입력해 주세요.")
            return

        st.session_state['sent_once'] = True

        with st.spinner("감정 분석 중..."):
            if backend == "KoBERT":
                res = extract_keywords_with_emotion_kobert(
                    sentence,
                    model,
                    tokenizer,
                    transform,
                    idx2label,
                    okt,
                    max_len=max_len
                )
            else:
                res = extract_keywords_with_emotion_koelectra(
                    sentence,
                    model,
                    tokenizer,
                    idx2label,
                    okt,
                    max_len=max_len
                )

        st.session_state['last_result'] = res

    # 2단계: 위쪽 채팅 영역 렌더
    with chat_container:
        # 1) 상대방 말풍선
        st.markdown("""
        <div class="chat-wrapper">
          <div class="chat-row">
            <div class="chat-bubble bot">
              오늘 어떤 일이 있었어?
            </div>
          </div>
        """, unsafe_allow_html=True)
    
        if st.session_state['last_result'] is not None:
            user_text = html.escape(st.session_state['last_result']["sentence"])
        else:
            user_text = '<span style="opacity:0.4;">여기에 당신의 메시지가 들어갑니다</span>'
    
        st.markdown(f"""
          <div class="chat-row">
            <div class="chat-bubble user">
              {user_text}
            </div>
          </div>
        </div> <!-- chat-wrapper end -->
        """, unsafe_allow_html=True)   

    # ===== 키보드 그림 =====
    if not st.session_state['sent_once']:
        try:
            st.image("keyboard.png", width=1000)
        except Exception:
            st.caption("💡 'keyboard.png' 파일을 같은 폴더에 두면 여기 키보드 그림이 뜹니다.")

    # ===== GIF 추천 =====
    if st.session_state['last_result'] is not None:
        res_kobert = st.session_state['last_result']

        st.markdown("---")
        st.markdown("### 🎬 GIF 추천")

        with st.spinner("GIPHY에서 GIF 검색 중..."):
            gif_urls, msg = recommend_gifs_from_result(
                res_kobert,
                giphy_api_key,
                limit_per_kw=3,
                total_limit=12
            )

        st.write(msg)

        if gif_urls:
            cols = st.columns(3)
            for i, url in enumerate(gif_urls):
                with cols[i % 3]:
                    st.image(url, width=250)

    # ===== 사이드바: 감정 분석 결과 영역 =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 감정 분석 결과")

    if st.session_state['last_result'] is not None:
        res = st.session_state['last_result']
        st.sidebar.markdown("**문장**")
        st.sidebar.write(res["sentence"])

        st.sidebar.markdown("**예측 감정**")
        st.sidebar.markdown(
            f"🧾 **{res['emotion_label']}**  "
            f"({res['emotion_prob']*100:.2f}% 확신)"
        )

        st.sidebar.markdown("**키워드**")
        if res["keywords"]:
            st.sidebar.write(", ".join(res["keywords"]))
        else:
            st.sidebar.write("_키워드가 추출되지 않았습니다._")
    else:
        st.sidebar.caption("아직 분석된 문장이 없습니다.")


if __name__ == "__main__":
    main()
