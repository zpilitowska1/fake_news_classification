# Fake News Classifier

Klasyfikator wiarygodności artykułów prasowych w języku polskim. Projekt obejmuje samodzielnie zebrany zbiór danych, pełny pipeline przetwarzania, trening modeli klasycznych i transformerowych oraz interfejs webowy.

## Spis treści

- [Dane treningowe](#dane-treningowe)
- [Preprocessing](#preprocessing)
- [Modele i wyniki](#modele-i-wyniki)
- [Interfejs](#interfejs)
- [Uruchomienie](#uruchomienie)
- [Struktura projektu](#struktura-projektu)

---

## Dane treningowe

Zbiór danych został zbudowany od podstaw przez scrapowanie trzech polskich źródeł, które łącznie dostarczyły ~5 600 artykułów po balansowaniu klas.

### Demagog.org.pl — klasa 1 (fake news)

Demagog to polska organizacja fact-checkingowa weryfikująca twierdzenia polityków i treści viralowe. Każdy artykuł w sekcji `/fake_news/` to zweryfikowany, fałszywy lub manipulacyjny przekaz — etykieta jest wbudowana w strukturę serwisu. Artykuły są zbierane przez RSS feed i obejmują różnorodne tematy: politykę, zdrowie, historię, gospodarkę. Często zawierają cytaty z mediów społecznościowych, posty z Facebooka czy Twittera oraz ich analizę przez ekspertów — co sprawia, że dane są bliskie realnym formatom dezinformacji.

Metoda: RSS feed (`/fake_news/feed/`), ~2 400 artykułów.

### OKO.press — klasa 0 (prawda)

OKO.press to niezależny portal dziennikarski specjalizujący się w reportażach śledczych i fact-checkingu. Artykuły są oparte na dokumentach, danych publicznych i wypowiedziach ekspertów. Wybrano kategorie tematyczne pokrywające się z tematyką Demagoga (polityka, prawa obywatelskie, gospodarka), co minimalizuje bias tematyczny między klasami. Artykuły mają zróżnicowaną formę: od krótkich newsów, przez analizy, po długie reportaże.

Metoda: scraping paginacji HTML z 9 kategorii tematycznych, ~2 300 artykułów.

### TVP Info — klasy 0 i 1

TVP Info dostarcza dwa rodzaje treści: sekcja `/sprawdzamy` to materiały fact-checkingowe (klasa 1, analogicznie do Demagoga), natomiast regularne newsy z sekcji informacyjnych stanowią klasę 0. Włączenie TVP Info zwiększa różnorodność stylistyczną danych — krótsze formy, inny styl narracji niż OKO.press.

Metoda: Selenium (dynamiczne ładowanie JavaScript), ~1 100 artykułów.

**Łącznie po balansowaniu klas (undersampling):** 4 529 train / 566 val / 566 test, podział stratyfikowany: 80/10/10

---

## Preprocessing

Surowe dane wymagały kilku etapów czyszczenia, których konieczność wynikła z analizy i obserwacji overfittingu modelu przy pierwszych iteracjach:

**Usunięcie markerów źródłowych.** Wstępne modele osiągały podejrzanie wysokie wyniki — analiza najważniejszych cech ujawniła, że model uczył się rozpoznawać źródła zamiast treści. Słowa takie jak `demagog`, `oko`, `press`, `sprawdzamy`, `tvp`, nazwiska autorów i redakcji były silnymi predykatorami prawdziwosci artykułu. Wszystkie takie markery zostały usunięte regexem.

**Usunięcie bloków RODO.** Artykuły z TVP Info zawierały na końcu pełną treść klauzul zgody RODO (`Kliknij Akceptuję i przechodzę do serwisu...`). Model mógłby nauczyć się korelacji między klasą 0 (TVP news) a obecnością bloku RODO. Tekst jest obcinany w miejscu wystąpienia tej frazy.

**Usunięcie biogramów autorów.** Artykuły OKO.press kończą się stałymi biogramami dziennikarzy (`Reporter, absolwent Polskiej Szkoły Reportażu...`). Podobnie jak markery źródłowe, były silnym wyznacznikiem klasy 0 niezwiązanym z treścią.

**Deduplikacja i normalizacja.** Usunięcie duplikatów po czyszczeniu, normalizacja whitespace, usunięcie HTML i URL-i.

**Sentyment jako cecha dodatkowa.** Do każdego artykułu dodano etykietę sentymentu (positive/neutral/negative) z modelu `cardiffnlp/twitter-xlm-roberta-base-sentiment` obsługującego język polski. Cecha ta jest używana jako dodatkowy sygnał w modelu baseline.

---

## Modele i wyniki

### Baseline — TF-IDF + modele klasyczne

Wektoryzacja TF-IDF (50 000 cech, unigramy i bigramy, `sublinear_tf=True`) z opcjonalnym one-hot sentymentu. Eksperymenty śledzone przez MLflow.

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.9823 | 0.9821 | 0.9928 | 0.9717 | 0.9980 |
| Logistic Regression C=1.0 | 0.9770 | 0.9768 | 0.9856 | 0.9682 | 0.9981 |
| Logistic Regression C=0.1 | 0.9452 | 0.9461 | 0.9315 | 0.9611 | 0.9916 |

Wybór modeli klasycznych jako baseline uzasadnia kilka czynników: szybki czas treningu, interpretowalność (widoczne wagi cech), brak wymagań sprzętowych.
XGBoost osiąga najwyższy F1 na zbiorze testowym, co potwierdza że zbiór danych jest wystarczająco duży i czysty by klasyczne metody działały skutecznie.

### HerBERT — fine-tuned transformer

Model `allegro/herbert-base-cased` — BERT wytrenowany przez Allegro na dużym korpusie polskiego tekstu. Fine-tuning na zebranym zbiorze danych (3 epoki, lr=2e-5, batch=16, max_len=256). Trening przeprowadziłam na GPU T4 (Google Colab).

| | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Val | 0.9717 | 0.9716 | 0.9751 | 0.9682 | 0.9943 |
| **Test** | **0.9735** | **0.9737** | **0.9653** | **0.9823** | **0.9951** |

HerBERT rozumie kontekst i znaczenie zdań, co jest szczególnie ważne przy tekście dziennikarskim gdzie ta sama informacja może być przedstawiona manipulacyjnie lub rzetelnie przy użyciu podobnych słów. Wysoki ROC-AUC (0.9951) wskazuje na bardzo dobrą separację klas.

**Wytrenowany model dostępny na Google Drive:**
[wyniki_herbert/model_final/ ](https://drive.google.com/drive/folders/16nxnJz5UwXP_5i87xlWleiSO8KdVbUM4?usp=sharing)

---

## Interfejs

Aplikacja webowa zbudowana w Streamlit.

![UI](https://github.com/ZofiaPilitowska240272/fake_news_classification/blob/main/docs/real_news_detection.png)
![UI](https://github.com/ZofiaPilitowska240272/fake_news_classification/blob/main/docs/fake_news%20words.png)

**Funkcje:**
- Wybór modelu (TF-IDF lub HerBERT) z opisem każdego
- Pole tekstowe do wklejenia artykułu
- Wynik klasyfikacji (FAKE NEWS / PRAWDA) z paskiem pewności modelu
- Dla modelu TF-IDF: wizualizacja kluczowych słów i zwrotów które wpłynęły na decyzję — czerwone wskazują na fake news, zielone na prawdę, z intensywnością proporcjonalną do wagi cechy
- Ostrzeżenie gdy pewność modelu spada poniżej 70%

---

## Uruchomienie

```bash
git clone https://github.com/ZofiaPilitowska240272/fake_news_classification.git
cd fake_news_classifier
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Opcja A — tylko TF-IDF

Dane i model są już w repozytorium (data/processed/), więc scraping i preprocessing można pominąć:
```bash
streamlit run app.py
```
Jeśli chcesz zebrać dane od nowa:

```bash
python src/data/scraper_demagog.py
python src/data/scraper_oko.py
python src/data/scraper_tvp.py
python src/data/preprocess.py
python src/models/baseline.py
streamlit run app.py
```

### Opcja B — z HerBERT

Pobierz wytrenowany model z Google Drive: [wyniki_herbert/model_final/ ](https://drive.google.com/drive/folders/16nxnJz5UwXP_5i87xlWleiSO8KdVbUM4?usp=sharing)  i wypakuj folder `model_final/` do `wyniki_herbert/model_final/`. Następnie uruchom aplikację jak w Opcji A.


### Testy

```bash
pytest tests/ -v
```

---

## Struktura projektu

```
fake_news_classifier/
├── docs/                        
├── data/
│   ├── raw/                    # surowe dane ze scraperów
│   └── processed/              # dane po preprocessingu
├── src/
│   ├── data/
│   │   ├── scraper_demagog.py
│   │   ├── scraper_oko.py
│   │   ├── scraper_tvp.py
│   │   └── preprocess.py
│   └── models/
│       ├── baseline.py
│       └── herbert_finetune.py
├── tests/
│   └── test_preprocess.py
├── wyniki_baseline/
├── wyniki_herbert/
├── app.py
├── requirements.txt
└── README.md
```
