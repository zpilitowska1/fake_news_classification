"""
Testy jednostkowe dla src/data/preprocess.py

Uruchom:
    pytest tests/ -v
"""

import sys
sys.path.insert(0, "src/data")

from preprocess import czyszczenie_bazowe, usun_markery, przygotuj_tekst


class TestCzyszczenieBlazowe:

    def test_usuwa_url(self):
        tekst = "Artykuł opublikowany na https://example.com jest fałszywy."
        wynik = czyszczenie_bazowe(tekst)
        assert "https://" not in wynik

    def test_usuwa_html(self):
        tekst = "Tekst z <b>tagiem HTML</b> i <a href='x'>linkiem</a>."
        wynik = czyszczenie_bazowe(tekst)
        assert "<b>" not in wynik
        assert "<a " not in wynik

    def test_usuwa_nowe_linie(self):
        tekst = "Pierwsza linia\nDruga linia\r\nTrzecia"
        wynik = czyszczenie_bazowe(tekst)
        assert "\n" not in wynik
        assert "\r" not in wynik

    def test_zamienia_cudzyslowy(self):
        tekst = 'Tekst z "cudzysłowem" w środku.'
        wynik = czyszczenie_bazowe(tekst)
        assert '"' not in wynik

    def test_pusty_tekst(self):
        assert czyszczenie_bazowe("") == ""

    def test_normalny_tekst(self):
        tekst = "Normalny artykuł bez żadnych problemów."
        wynik = czyszczenie_bazowe(tekst)
        assert wynik == tekst


class TestUsunMarkery:

    def test_usuwa_pisownia_oryginalna(self):
        tekst = "Internauci pisali [pisownia oryginalna]: warto przeczytać."
        wynik = usun_markery(tekst)
        assert "pisownia oryginalna" not in wynik.lower()

    def test_usuwa_czas_nagrania(self):
        tekst = "W nagraniu czas nagrania 0:45 słyszymy twierdzenie."
        wynik = usun_markery(tekst)
        assert "czas nagrania" not in wynik.lower()

    def test_usuwa_demagog(self):
        tekst = "Jak pisał demagog.org.pl w poprzednim artykule."
        wynik = usun_markery(tekst)
        assert "demagog.org.pl" not in wynik.lower()

    def test_usuwa_oko_press(self):
        tekst = "Jak donosi oko.press, sprawa trafiła do sądu."
        wynik = usun_markery(tekst)
        assert "oko.press" not in wynik.lower()

    def test_usuwa_tvp(self):
        tekst = "Program tvp.info pokazał materiał."
        wynik = usun_markery(tekst)
        assert "tvp.info" not in wynik.lower()

    def test_pusty_tekst(self):
        assert usun_markery("") == ""


class TestPrzygotujTekst:

    def test_usuwa_rodo(self):
        tekst = "Treść artykułu. Kliknij Akceptuję i przechodzę do serwisu dane osobowe."
        wynik = przygotuj_tekst(tekst)
        assert "Akceptuję" not in wynik
        assert "Treść artykułu" in wynik

    def test_usuwa_biogram_reportera(self):
        tekst = "Ważny artykuł o polityce. Reporter, absolwent Polskiej Szkoły Reportażu."
        wynik = przygotuj_tekst(tekst)
        assert "absolwent" not in wynik
        assert "artykuł o polityce" in wynik

    def test_pelny_pipeline(self):
        tekst = "Artykuł z https://example.com i <b>tagiem</b>.\nNowa linia."
        wynik = przygotuj_tekst(tekst)
        assert "https://" not in wynik
        assert "<b>" not in wynik
        assert "\n" not in wynik

    def test_pusty_tekst(self):
        assert przygotuj_tekst("") == ""