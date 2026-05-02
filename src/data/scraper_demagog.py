"""
Scraper demagog.org.pl — zbiera artykuły z kategorii /fake_news/ przez RSS.
Etykieta 1 = fake news

python src/data/scraper_demagog.py
"""

import requests
import pandas as pd
import time
import random
import html as html_module
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

BASE     = "https://demagog.org.pl"
FEED_URL = f"{BASE}/fake_news/feed/"
OUTPUT   = Path("data/raw/demagog_final.csv")

FRAZY_STOP = [
    "Wyrażam zgodę", "przetwarzanie moich danych", "Stowarzyszenie Demagog",
    "polityce prywatności", "Zapisz się", "newsletter",
    "Oferta szkoleń", "Copyright", "Jeśli znajdziesz błąd",
    "Szybkie alerty", "Oddzielamy prawdę", "Patrz władzy",
]


def parsuj_rss(xml_bytes: bytes) -> list[str]:
    linki = []
    try:
        root = ET.fromstring(xml_bytes)
        channel = root.find("channel")
        items = channel.findall("item") if channel is not None else root.findall(".//item")

        for item in items:
            link_el = item.find("link")
            guid_el = item.find("guid")
            link = (
                (link_el.text.strip() if link_el is not None and link_el.text else None)
                or (guid_el.text.strip() if guid_el is not None and guid_el.text
                    and guid_el.text.startswith("http") else None)
            )
            if link:
                linki.append(link)
    except ET.ParseError as e:
        print(f"  XML błąd: {e}")
    return linki


def zbierz_linki(max_stron: int = 100) -> list[str]:
    wszystkie = set()
    poprzednia = set()

    for page in range(1, max_stron + 1):
        url = FEED_URL if page == 1 else f"{FEED_URL}?paged={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            if res.status_code != 200:
                print(f"  Strona {page}: HTTP {res.status_code} — kończę")
                break

            ta_strona = set(parsuj_rss(res.content))
            nowe = ta_strona - wszystkie
            wszystkie.update(ta_strona)
            print(f"  Strona {page}: +{len(nowe)} (łącznie: {len(wszystkie)})")

            if not ta_strona or ta_strona == poprzednia:
                print("  Brak nowych — koniec")
                break

            poprzednia = ta_strona

        except Exception as e:
            print(f"  Strona {page}: błąd — {e}")
            break

        time.sleep(random.uniform(0.8, 1.5))

    return list(wszystkie)


def pobierz_tekst(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")
        kontener = (
            soup.find("article") or soup.find("main")
            or soup.find(class_=lambda c: c and any(
                x in " ".join(c).lower()
                for x in ["content", "article", "post", "entry", "body"]
            ))
        )
        paragrafy = (kontener or soup).find_all("p")
        czysty = [
            html_module.unescape(p.get_text(separator=" ", strip=True))
            for p in paragrafy
            if len(p.get_text(strip=True)) >= 80
            and not any(f.lower() in p.get_text().lower() for f in FRAZY_STOP)
        ]
        return " ".join(czysty)

    except Exception as e:
        print(f"  błąd: {e}")
        return ""


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Zbieram linki z RSS: {FEED_URL}")
    linki = zbierz_linki()
    print(f"\n→ {len(linki)} unikalnych artykułów\n")

    zebrane = []
    for i, url in enumerate(linki, 1):
        print(f"  [{i:4d}/{len(linki)}] {url}")
        tekst = pobierz_tekst(url)
        if tekst:
            zebrane.append({
                "Tresc": tekst,
                "Etykieta": 1,
                "Kategoria": "fake_news",
                "Zrodlo": url,
            })
        time.sleep(random.uniform(1.0, 2.0))

    df = pd.DataFrame(zebrane)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nZapisano {len(df)} artykułów")
    print(f"Mediana długości: {int(df['Tresc'].str.len().median())} znaków")


if __name__ == "__main__":
    main()