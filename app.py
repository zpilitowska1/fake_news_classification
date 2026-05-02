"""
UI: Streamlit app — detektor fake news.

Uruchomienie:
    streamlit run app.py
"""

import streamlit as st
import re
import time
from pathlib import Path

st.set_page_config(
    page_title="Detektor Fake News",
    page_icon="",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #f8f7f4; color: #1a1a1a; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    letter-spacing: -1px; color: #1a1a1a; margin-bottom: 0.2rem;
    white-space: nowrap;
}
.subtitle {
    color: #888; font-size: 0.9rem; margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
}
.model-desc {
    background: #fff; border: 1px solid #e5e5e5; border-radius: 10px;
    padding: 1rem 1.2rem; margin: 0.5rem 0 1.5rem;
    font-size: 0.88rem; color: #555; line-height: 1.6;
}
.model-desc strong { color: #1a1a1a; font-weight: 600; }
.model-desc .badge {
    display: inline-block; background: #eef2ff; color: #4338ca;
    border-radius: 99px; padding: 0.1rem 0.6rem;
    font-size: 0.75rem; font-family: 'Space Mono', monospace;
    margin-left: 0.4rem;
}
.loading-wrap {
    text-align: center; padding: 2.5rem 0;
    font-family: 'Space Mono', monospace; color: #888; font-size: 0.85rem;
}
.loading-dot {
    display: inline-block; width: 8px; height: 8px;
    background: #4338ca; border-radius: 50%; margin: 0 3px;
    animation: bounce 1s infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.15s; }
.loading-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
    0%,100% { transform: translateY(0); opacity: 0.4; }
    50% { transform: translateY(-6px); opacity: 1; }
}
.result-box {
    border-radius: 14px; padding: 2rem;
    margin: 1.5rem 0; text-align: center;
    animation: fadeIn 0.4s ease;
}
.result-fake {
    background: #fff5f5; border: 1.5px solid #fc8181;
    box-shadow: 0 2px 16px rgba(252,129,129,0.12);
}
.result-real {
    background: #f0fff4; border: 1.5px solid #68d391;
    box-shadow: 0 2px 16px rgba(104,211,145,0.12);
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    letter-spacing: 2px; margin-bottom: 0.5rem;
}
.result-fake .result-label { color: #e53e3e; }
.result-real .result-label { color: #276749; }
.confidence-bar-wrap {
    background: #eee; border-radius: 99px; height: 8px;
    margin: 0.8rem auto; max-width: 300px; overflow: hidden;
}
.confidence-bar-inner { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
.confidence-label { font-size: 0.83rem; color: #666; font-family: 'Space Mono', monospace; }
.detail-chip {
    display: inline-block; background: #f0f0f0; border: 1px solid #e0e0e0;
    border-radius: 99px; padding: 0.25rem 0.75rem;
    font-size: 0.76rem; color: #666; margin: 0.2rem;
    font-family: 'Space Mono', monospace;
}
.warning-box {
    background: #fffbeb; border: 1px solid #f6ad55; border-radius: 8px;
    padding: 0.8rem 1rem; color: #92400e; font-size: 0.85rem; margin-top: 1rem;
}
.stAlert { color: #92400e !important; }
.divider { border: none; border-top: 1px solid #e5e5e5; margin: 2rem 0; }

textarea {
    background: #eef4ff !important;
    border: 1.5px solid #c7d9f8 !important;
    color: #1a1a1a !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus {
    border-color: #4338ca !important;
    box-shadow: 0 0 0 3px rgba(67,56,202,0.1) !important;
}
.stButton > button {
    background: #1a1a1a; color: #f8f7f4;
    font-family: 'Space Mono', monospace; font-weight: 700;
    letter-spacing: 1px; border: none; border-radius: 8px;
    padding: 0.7rem 2rem; font-size: 0.9rem;
    transition: all 0.2s; width: 100%;
}
.stButton > button:hover { background: #333; transform: translateY(-1px); }

.stExpander {
    background: #f0eeea !important;
    border: 1px solid #e0ddd8 !important;
    border-radius: 10px !important;
}
.stExpander summary {
    color: #555 !important;
    font-size: 0.9rem !important;
}
.stRadio label, .stRadio div[data-testid="stMarkdownContainer"] p,
.stRadio span, .stRadio div {
    color: #1a1a1a !important;
    font-size: 0.95rem !important;
}
.stRadio > div { gap: 0.5rem; }
.stRadio label, .stRadio div[data-testid="stMarkdownContainer"] p,
.stRadio span, [data-testid="stRadio"] label span {
    color: #1a1a1a !important;
    font-size: 0.95rem !important;
}
.stRadio > div { gap: 0.5rem; }

</style>
""", unsafe_allow_html=True)

# ── MARKERY ────────────────────────────────────────────────────
MARKERY = [
    r"pisownia oryginalna", r"czas nagrania\s*\d+:\d+", r"demagog\.org\.pl",
    r"\bdemagog\b", r"\b(facebook|twitter|tiktok|youtube|instagram|x\.com)\b",
    r"w mediach spolecznosciowych", r"\boko\b", r"\bpress\b", r"oko\.press",
    r"gazet[ae]\s*wyborczej?", r"gazeta\s*wyborcza", r"\bwyborczej\b",
    r"\btvp\b", r"tvp\.info", r"\bsprawdzamy\b",
    r"juliusz\s*gluski[\-\s]*schimmer", r"witold\s*tabaka",
    r"\bgluski\b", r"\bschimmer\b", r"\btabaka\b",
]
MARKERY_RE = re.compile("|".join(MARKERY), flags=re.IGNORECASE)
RODO_RE    = re.compile(r"kliknij\s+.{0,30}akceptuj", flags=re.IGNORECASE)


def przygotuj_tekst(tekst: str) -> str:
    tekst = re.sub(r"https?://\S+", " ", tekst)
    tekst = re.sub(r"<[^>]+>", " ", tekst)
    tekst = re.sub(r"[\r\n\t]+", " ", tekst)
    m = RODO_RE.search(tekst)
    if m:
        tekst = tekst[:m.start()]
    tekst = MARKERY_RE.sub(" ", tekst)
    return re.sub(r"\s+", " ", tekst).strip()


# ladowanie modelu
@st.cache_resource(show_spinner=False)
def zaladuj_herbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_path = "wyniki_herbert/model_final"
    if not Path(model_path).exists():
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def zaladuj_tfidf():
    import pickle
    pkl = Path("wyniki_baseline/lr_model.pkl")
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    proc = Path("data/processed")
    if not (proc / "train_clean.csv").exists():
        return None
    train = pd.read_csv(proc / "train_clean.csv", encoding="utf-8-sig")
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                             sublinear_tf=True, min_df=2)
    X = tfidf.fit_transform(train["Tresc"].fillna(""))
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X, train["Etykieta"])
    return tfidf, clf


def predykcja_herbert(tekst, tokenizer, model):
    import torch
    inputs = tokenizer(tekst, truncation=True, max_length=256,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = int(probs.argmax())
    return float(probs[pred]), pred


def predykcja_tfidf(tekst, tfidf, clf):
    vec  = tfidf.transform([tekst])
    pred = int(clf.predict(vec)[0])
    prob = float(clf.predict_proba(vec)[0][pred])
    return prob, pred


def top_slowa_tfidf(tekst: str, tfidf, clf, n: int = 10):
    """Zwraca top n slow ktore najbardziej wskazuja na fake/prawda."""
    import numpy as np
    vec = tfidf.transform([tekst])
    nazwy = tfidf.get_feature_names_out()
    wagi_modelu = clf.coef_[0]  
    indeksy = vec.nonzero()[1]
    wyniki = []
    for i in indeksy:
        tfidf_val = vec[0, i]
        wklad = float(wagi_modelu[i] * tfidf_val)
        wyniki.append((nazwy[i], wklad))

    wyniki.sort(key=lambda x: abs(x[1]), reverse=True)
    fake_slowa  = [(s, w) for s, w in wyniki if w > 0][:n]
    prawda_slowa = [(s, w) for s, w in wyniki if w < 0][:n]
    return fake_slowa, prawda_slowa


# UI
st.markdown('<div class="main-title">// FAKE NEWS DETEKTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">weryfikacja wiarygodnosci tekstu · polski NLP</div>', unsafe_allow_html=True)

# Ustawienia zaawansowane
with st.expander("Wybór modelu", expanded=False):
    tryb = st.radio(
        "Wybierz model",
        ["TF-IDF + Logistic Regression", "HerBERT (fine-tuned)"],
        label_visibility="collapsed",
    )
    opisy = {
        "TF-IDF + Logistic Regression": (
            "<strong>TF-IDF + Logistic Regression</strong>"
            "<span class='badge'>szybki</span><br>"
            "Zamienia tekst na wektory częstości słów i klasyfikuje regresją logistyczną. "
            "Działa bez GPU, odpowiedź w ułamku sekundy."
        ),
        "HerBERT (fine-tuned)": (
            "<strong>HerBERT fine-tuned</strong>"
            "<span class='badge'>dokładniejszy</span><br>"
            "Transformer wytrenowany na polskim tekście, dostrojony na danych z Demagog, "
            "OKO.press i TVP Info. Rozumie kontekst i znaczenie zdań."
        ),
    }
    st.markdown(f'<div class="model-desc">{opisy[tryb]}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:Space Mono,monospace;font-size:0.82rem;"
    "color:#888;margin-bottom:0.4rem;'>Wklej tekst do analizy</p>",
    unsafe_allow_html=True
)
tekst_input = st.text_area(
    "Tekst",
    placeholder="Wklej tutaj artykul, post lub fragment newsa po polsku...",
    height=240,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analizuj = st.button("ANALIZUJ", use_container_width=True)

# Wynik:
if analizuj:
    if not tekst_input.strip():
        st.markdown('<div style="background:#fffbeb;border:1px solid #f6ad55;border-radius:8px;padding:0.8rem 1rem;color:#92400e !important;font-size:0.88rem;margin-top:0.5rem;">Wklej tekst przed analizą.</div>', unsafe_allow_html=True)
    elif len(tekst_input.strip()) < 50:
        st.markdown('<div style="background:#fffbeb;border:1px solid #f6ad55;border-radius:8px;padding:0.8rem 1rem;color:#92400e !important;font-size:0.88rem;margin-top:0.5rem;">Tekst za krótki — wklej przynajmniej kilka zdań.</div>', unsafe_allow_html=True)
    else:
        ladowanie = st.empty()
        ladowanie.markdown("""
        <div class="loading-wrap">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div style="margin-top:0.8rem">Analizuję tekst...</div>
        </div>
        """, unsafe_allow_html=True)

        tekst_czysty = przygotuj_tekst(tekst_input)
        prob, pred   = None, None
        blad         = None
        tfidf, clf   = None, None

        try:
            if tryb == "HerBERT (fine-tuned)":
                tokenizer, model = zaladuj_herbert()
                if tokenizer is None:
                    blad = "Nie znaleziono modelu HerBERT w <code>wyniki_herbert/model_final/</code>."
                else:
                    prob, pred = predykcja_herbert(tekst_czysty, tokenizer, model)
            else:
                wynik = zaladuj_tfidf()
                if wynik is None:
                    blad = "Brak danych w <code>data/processed/</code>. Uruchom najpierw <code>preprocess.py</code>."
                else:
                    tfidf, clf = wynik
                    prob, pred = predykcja_tfidf(tekst_czysty, tfidf, clf)
        except Exception as e:
            blad = f"Błąd modelu: {e}"

        ladowanie.empty()

        if blad:
            st.error(blad, icon="⚠️")
        else:
            is_fake   = pred == 1
            css_class = "result-fake" if is_fake else "result-real"
            label     = "FAKE NEWS" if is_fake else "PRAWDA"
            bar_color = "#fc8181" if is_fake else "#68d391"
            bar_pct   = int(prob * 100)

            st.markdown(f"""
            <div class="result-box {css_class}">
                <div class="result-label">{label}</div>
                <div class="confidence-label">pewnosc modelu: {bar_pct}%</div>
                <div class="confidence-bar-wrap">
                    <div class="confidence-bar-inner"
                         style="width:{bar_pct}%; background:{bar_color};"></div>
                </div>
                <span class="detail-chip">{len(tekst_input.split())} slow</span>
                <span class="detail-chip">{len(tekst_input)} znakow</span>
                <span class="detail-chip">{tryb.split()[0]}</span>
            </div>
            """, unsafe_allow_html=True)

            if prob < 0.7:
                st.markdown(
                    '<div style="background:#fffbeb;border:1px solid #f6ad55;border-radius:8px;'
                    'padding:0.8rem 1rem;color:#92400e;font-size:0.85rem;margin-top:1rem;">'
                    'Wynik niejednoznaczny — pewność modelu poniżej 70%. '
                    'Zweryfikuj tekst w dodatkowych źródłach.</div>',
                    unsafe_allow_html=True
                )

            # Kluczowe slowa — tylko dla TF-IDF
            if tryb == "TF-IDF + Logistic Regression" and tfidf is not None:
                fake_slowa, prawda_slowa = top_slowa_tfidf(
                    tekst_czysty, tfidf, clf, n=10
                )
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-family:Space Mono,monospace;font-size:0.82rem;"
                    "color:#888;margin-bottom:0.8rem;'>Kluczowe słowa w tekście</p>",
                    unsafe_allow_html=True
                )
                if is_fake:
                    st.markdown(
                        "<p style='font-size:0.8rem;color:#e53e3e;font-weight:600;"
                        "font-family:Space Mono,monospace;margin-bottom:0.6rem;'>"
                        "Słowa wskazujące na FAKE NEWS</p>",
                        unsafe_allow_html=True
                    )
                    for slowo, waga in fake_slowa:
                        st.markdown(
                            f"<span style='display:inline-block;"
                            f"background:rgba(252,129,129,{min(abs(waga)*3, 0.9):.2f});"
                            f"border-radius:6px;padding:0.2rem 0.6rem;margin:0.15rem;"
                            f"font-size:0.85rem;color:#1a1a1a;'>{slowo}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<p style='font-size:0.8rem;color:#276749;font-weight:600;"
                        "font-family:Space Mono,monospace;margin-bottom:0.6rem;'>"
                        "Słowa wskazujące na PRAWDĘ</p>",
                        unsafe_allow_html=True
                    )
                    for slowo, waga in prawda_slowa:
                        st.markdown(
                            f"<span style='display:inline-block;"
                            f"background:rgba(104,211,145,{min(abs(waga)*3, 0.9):.2f});"
                            f"border-radius:6px;padding:0.2rem 0.6rem;margin:0.15rem;"
                            f"font-size:0.85rem;color:#1a1a1a;'>{slowo}</span>",
                            unsafe_allow_html=True
                        )

#Stopka
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#aaa;font-size:0.75rem;font-family:Space Mono,monospace;">'
    'Model trenowany na danych z Demagog · OKO.press · TVP Info'
    '</div>',
    unsafe_allow_html=True
)