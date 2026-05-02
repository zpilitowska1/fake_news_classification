"""
Scraper TVP Info z Selenium:
- tvp.info/sprawdzamy -> etykieta 1 (fake news)
- tvp.info newsy      -> etykieta 0 (prawda)

    python src/data/scraper_tvp.py
"""

import time
import random
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

BASE   = "https://www.tvp.info"
OUTPUT = Path("data/raw/tvp_final.csv")

FRAZY_STOP = [
    "Polityka prywatnosci", "Regulamin", "Copyright", "Reklama",
    "Zapisz sie", "newsletter", "Udostepnij", "Komentarze",
    "Czytaj wiecej", "Powiazane artykuly", "Polecamy rowniez",
    "TVP SA", "Telewizja Polska", "tvp.info",
]

SEKCJE_PRAWDA = [
    f"{BASE}/polska", f"{BASE}/swiat",
    f"{BASE}/gospodarka", f"{BASE}/spoleczenstwo", f"{BASE}/nauka",
]


def zrob_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def czy_artykul(href: str, wyklucz_id: str = None) -> bool:
    if not href.startswith(BASE):
        return False
    sciezka = href.replace(BASE, "").strip("/")
    segmenty = sciezka.split("/")
    if len(segmenty) < 2 or not segmenty[0].isdigit():
        return False
    if wyklucz_id and segmenty[0] == wyklucz_id:
        return False
    if any(x in href for x in ["/tag/", "/autor/", "#", "?", "javascript"]):
        return False
    return True


def zbierz_linki_ze_strony(driver, url: str, wyklucz_id: str = None) -> set:
    driver.get(url)
    time.sleep(random.uniform(2, 3))
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, 800)")
        time.sleep(0.5)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    linki = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = BASE + href
        if czy_artykul(href, wyklucz_id):
            linki.add(href)
    return linki


def zbierz_linki_kategorii(driver, url_baza: str, max_stron: int = 60,
                            wyklucz_id: str = None, limit: int = 550) -> list[str]:
    wszystkie = set()
    for page in range(1, max_stron + 1):
        url = url_baza if page == 1 else f"{url_baza}?page={page}"
        print(f"  Strona {page}", end=" -> ")
        try:
            nowe = zbierz_linki_ze_strony(driver, url, wyklucz_id) - wszystkie
            wszystkie.update(nowe)
            print(f"+{len(nowe)} (lacznie: {len(wszystkie)})")
            if len(nowe) == 0 and page > 1:
                print("  Brak nowych")
                break
        except Exception as e:
            print(f"blad: {e}")
            break
        time.sleep(random.uniform(1.5, 2.5))
        if len(wszystkie) >= limit:
            break
    return list(wszystkie)


def pobierz_tekst(driver, url: str) -> str:
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 2.5))
        soup = BeautifulSoup(driver.page_source, "html.parser")
        kontener = (
            soup.find("div", class_=lambda c: c and any(
                x in " ".join(c).lower()
                for x in ["article-body", "article__body", "article__content",
                           "content-body", "entry-content"]
            ))
            or soup.find("article")
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
    driver = zrob_driver()
    zebrane = []

    try:
        print("\nTVP Sprawdzamy (etykieta=1): ")
        linki_fake = zbierz_linki_kategorii(
            driver, f"{BASE}/77454336/sprawdzamy",
            max_stron=60, wyklucz_id="77454336", limit=550,
        )[:550]
        print(f"\n-> {len(linki_fake)} artykulow\n")

        for i, url in enumerate(linki_fake, 1):
            print(f"  [{i:3d}/{len(linki_fake)}] {url}")
            tekst = pobierz_tekst(driver, url)
            if tekst:
                zebrane.append({"Tresc": tekst, "Etykieta": 1,
                                 "Kategoria": "tvp_sprawdzamy", "Zrodlo": url})
            else:
                print("         pusty")

        print("\nTVP Info newsy (etykieta=0): ")
        linki_prawda: set = set()

        for sekcja in SEKCJE_PRAWDA:
            print(f"\n[{sekcja.split('/')[-1]}]")
            nowe = zbierz_linki_kategorii(
                driver, sekcja, max_stron=12, wyklucz_id="77454336", limit=550
            )
            linki_prawda.update(nowe)
            print(f"  Lacznie: {len(linki_prawda)}")
            if len(linki_prawda) >= 550:
                break

        for i, url in enumerate(list(linki_prawda)[:550], 1):
            print(f"  [{i:3d}/550] {url}")
            tekst = pobierz_tekst(driver, url)
            if tekst:
                zebrane.append({"Tresc": tekst, "Etykieta": 0,
                                 "Kategoria": "tvp_news", "Zrodlo": url})
            else:
                print("         pusty")

    finally:
        driver.quit()

    df = pd.DataFrame(zebrane)
    df = df[df["Tresc"].str.len() >= 300]
    print(f"\nPodsumowanie: ")
    print(df.groupby(["Kategoria", "Etykieta"]).size().to_string())
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"Zapisano -> {OUTPUT}")


if __name__ == "__main__":
    main()