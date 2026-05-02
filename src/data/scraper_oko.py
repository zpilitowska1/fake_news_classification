"""
Scraper oko.press — zbiera artykuły z zakadek: /temat/ i /kategoria/
Etykieta 0 = prawda.

    python src/data/scraper_oko.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "pl-PL,pl;q=0.9",
    "Referer": "https://oko.press/",
}

BASE   = "https://oko.press"
OUTPUT = Path("data/raw/oko_press_final.csv")

FRAZY_STOP = [
    "Zapisz się", "newsletter", "Wesprzyj OKO", "Przekaż darowiznę",
    "Copyright", "Polityka prywatności", "Regulamin",
    "Czytaj więcej", "Przeczytaj także", "Reklama",
]

TEMATY = [
    "gospodarka", "sadownictwo", "wybory",
    "wladza", "media",
]

KATEGORIE = [
    "sledztwa", "analizy-sondaze", "rozmowa", "reportaz",
]

ZRODLA = (
    [(f"{BASE}/temat/{s}/", s) for s in TEMATY]
    + [(f"{BASE}/kategoria/{s}/", s) for s in KATEGORIE]
)


def zbierz_linki(url_baza: str, slug: str, max_stron: int = 20) -> list[str]:
    linki = set()
    url_baza_clean = url_baza.rstrip("/")

    for page in range(1, max_stron + 1):
        url = url_baza if page == 1 else f"{url_baza_clean}?page={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=12)
            if res.status_code != 200:
                break

            soup = BeautifulSoup(res.text, "html.parser")
            nowe = 0

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/"):
                    href = BASE + href

                sciezka = href.replace(BASE, "").strip("/")
                segmenty = sciezka.split("/")
                if (
                    href.startswith(BASE + "/")
                    and len(segmenty) == 1
                    and len(segmenty[0]) > 10
                    and "#" not in href
                    and "?" not in href
                    and href not in linki
                ):
                    linki.add(href)
                    nowe += 1

            print(f"    Strona {page}: +{nowe} (lacznie: {len(linki)})")

            if nowe == 0 and page > 1:
                break

        except Exception as e:
            print(f"    Blad strona {page}: {e}")
            break

        time.sleep(random.uniform(0.8, 1.5))

    return list(linki)


def pobierz_tekst(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=12)
        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")
        kontener = (
            soup.find("article")
            or soup.find(class_=lambda c: c and any(
                x in " ".join(c).lower()
                for x in ["entry-content", "post-content", "article__body",
                           "article-body", "content-body"]
            ))
            or soup.find("main")
        )
        paragrafy = (kontener or soup).find_all("p")
        czysty = [
            p.get_text(separator=" ", strip=True)
            for p in paragrafy
            if len(p.get_text(strip=True)) >= 80
            and not any(f.lower() in p.get_text().lower() for f in FRAZY_STOP)
        ]
        return " ".join(czysty)

    except Exception as e:
        print(f"    blad: {e}")
        return ""


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print("Zbieranie linkow: \n")
    wszystkie = set()

    for url_baza, slug in ZRODLA:
        print(f"[{slug}]")
        linki = zbierz_linki(url_baza, slug)
        wszystkie.update(linki)
        print(f"   {len(linki)} | lacznie: {len(wszystkie)}\n")

    linki_lista = list(wszystkie)
    print(f"Unikalnych artykulow: {len(linki_lista)}")

    print("\nPobieranie tresci ")
    zebrane = []

    for i, url in enumerate(linki_lista, 1):
        print(f"  [{i:4d}/{len(linki_lista)}] {url}")
        tekst = pobierz_tekst(url)
        if tekst:
            zebrane.append({
                "Tresc": tekst,
                "Etykieta": 0,
                "Kategoria": "oko.press",
                "Zrodlo": url,
            })
        else:
            print("         pusty")

        time.sleep(random.uniform(1.2, 2.5))

    df = pd.DataFrame(zebrane)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\nZapisano {len(df)} artykulow do: {OUTPUT}")


if __name__ == "__main__":
    main()