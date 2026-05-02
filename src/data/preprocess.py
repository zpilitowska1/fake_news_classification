"""
Preprocessing — laczy dane z 3 scraperow, czysci tekst, usuwa markery zrodlowe, dodaje sentyment, dzieli na train/val/test.

    python src/data/preprocess.py

"""

import re
import csv
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import pipeline

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

PLIKI = {
    "demagog":  RAW_DIR / "demagog_final.csv",
    "oko":      RAW_DIR / "oko_press_final.csv",
    "tvp":      RAW_DIR / "tvp_final.csv",
}

MARKERY = [
    r"obalamy fa[łl]szywe informacje pojawi[ae]j[aą]ce si[eę] w mediach[^.]*\.",
    r"odwo[łl]uj[aą]c si[eę] do wiarygodnych [źz]r[oó]de[łl][^.]*weryfikujemy[^.]*\.",
    r"najbardziej szkodliwe przyk[łl]ady dezinformacji",
    r"pisownia oryginalna", r"czas nagrania\s*\d+:\d+", r"demagog\.org\.pl",
    r"\bdemagog\b",
    r"\b(facebook|twitter|tiktok|youtube|instagram|x\.com)\b",
    r"w mediach spo[łl]eczno[śs]ciowych",
    r"\boko\b", r"\bpress\b", r"oko\.press",
    r"absolwentka? [^.]{0,60}uniwersytetu[^.]*\.",
    r"publikowa[łl][ao]? m\.in\.[^.]*\.",
    r"gazet[ae]\s*wyborczej?", r"gazeta\s*wyborcza", r"\bwyborczej\b",
    r"\btvp\b", r"tvp\.info", r"\bsprawdzamy\b",
    r"juliusz\s*gluski[\-\s]*schimmer", r"witold\s*tabaka",
    r"\bgluski\b", r"\bschimmer\b", r"\btabaka\b",
]

# Blok tekstu dotyczcy RODO
RODO_WZORZEC = re.compile(
    r"kliknij\s+.{0,30}akceptuj",
    flags=re.IGNORECASE,
)

# Obcina biogramy autorow OKO.press
BIOGRAM_WZORZEC = re.compile(
    r"(reporter,?\s+absolwent|dziennikarz,?\s+publicysta,?\s+rocznik"
    r"|absolwentka?\s+prawa|za cykl reporta[żz]|wspó[łl]za[łl]o[żź]y[łl])",
    flags=re.IGNORECASE,
)
MARKERY_WZORZEC = re.compile("|".join(MARKERY), flags=re.IGNORECASE)


def usun_markery(tekst: str) -> str:
    return MARKERY_WZORZEC.sub(" ", tekst)


def czyszczenie_bazowe(tekst: str) -> str:
    tekst = re.sub(r"https?://\S+", " ", tekst)
    tekst = re.sub(r"<[^>]+>", " ", tekst)
    tekst = re.sub(r"[\r\n\t]+", " ", tekst)
    tekst = re.sub(r'"', "'", tekst)
    tekst = re.sub(r"\s+", " ", tekst).strip()
    return tekst


def przygotuj_tekst(tekst: str) -> str:
    tekst = czyszczenie_bazowe(tekst)
    m = RODO_WZORZEC.search(tekst)
    if m:
        tekst = tekst[:m.start()]
    m = BIOGRAM_WZORZEC.search(tekst)
    if m:
        tekst = tekst[:m.start()]
    tekst = usun_markery(tekst)
    tekst = re.sub(r"\s+", " ", tekst).strip()
    return tekst


def wczytaj_dane() -> pd.DataFrame:
    ramki = []
    for nazwa, sciezka in PLIKI.items():
        if not sciezka.exists():
            print(f"  Brak pliku: {sciezka}, pomija")
            continue
        df = pd.read_csv(sciezka, encoding="utf-8-sig")
        df = df.dropna(subset=["Tresc", "Etykieta"])
        df["Etykieta"] = df["Etykieta"].astype(int)
        print(f"  {nazwa}: {len(df)} rekordow")
        ramki.append(df[["Tresc", "Etykieta", "Kategoria", "Zrodlo"]])
    return pd.concat(ramki, ignore_index=True)


def dodaj_sentyment(df: pd.DataFrame) -> pd.DataFrame:
    print("\nLadowanie modelu...")
    analizator = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        truncation=True,
        max_length=512,
        batch_size=64,
        device=-1,
    )

    print(f"Analizuje sentyment dla {len(df)} tekstow...")
    teksty = df["Tresc"].str[:1000].tolist()
    wyniki = analizator(teksty)

    mapa = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    df["Sentyment"] = [mapa.get(w["label"], w["label"]) for w in wyniki]
    return df


def balansuj(df: pd.DataFrame) -> pd.DataFrame:
    min_n = df["Etykieta"].value_counts().min()
    return (
        df.groupby("Etykieta", group_keys=False)
          .apply(lambda x: x.sample(min_n, random_state=42))
          .reset_index(drop=True)
    )


def podziel(df: pd.DataFrame):
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df["Etykieta"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["Etykieta"], random_state=42
    )
    return train, val, test


def main():
    df = wczytaj_dane()
    print(f"Wczytano: {len(df)} rekordow")

    df["Tresc"] = df["Tresc"].astype(str).apply(przygotuj_tekst)
    przed = len(df)
    df = df[df["Tresc"].str.len() > 0].reset_index(drop=True)
    df = df.drop_duplicates(subset=["Tresc"], keep="first").reset_index(drop=True)
    print(f"Po czyszczeniu: {len(df)} (usunieto {przed - len(df)} pustych/duplikatow)")

    print("\nRozklad etykiet przed balansowaniem:")
    print(df["Etykieta"].value_counts().to_string())

    df = dodaj_sentyment(df)
    print("\nRozklad sentymentu:")
    print(df["Sentyment"].value_counts().to_string())

    print("\nSentyment vs etykieta:")
    print(pd.crosstab(df["Etykieta"], df["Sentyment"]).to_string())

    df = balansuj(df)
    print(f"Po balansowaniu: {len(df)}")
    print(df["Etykieta"].value_counts().to_string())

    print("\nPodzial danych: ")
    train, val, test = podziel(df)
    print(f"Train: {len(train)}   Val: {len(val)}   Test: {len(test)}")

    save_kw = dict(index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    df.to_csv(PROC_DIR / "dataset_czysty.csv",  **save_kw)
    train.to_csv(PROC_DIR / "train_clean.csv",  **save_kw)
    val.to_csv(PROC_DIR   / "val_clean.csv",    **save_kw)
    test.to_csv(PROC_DIR  / "test_clean.csv",   **save_kw)

    print("\nZapisano pliki do data/processed/")


if __name__ == "__main__":
    main()