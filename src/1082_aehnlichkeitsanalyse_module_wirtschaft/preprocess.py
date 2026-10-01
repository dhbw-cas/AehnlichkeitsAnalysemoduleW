"""Bereitet `module.csv` für die Ähnlichkeitsanalyse auf (Embeddings + Literaturabgleich).

Schritte (deterministisch, ohne externe Dienste):
1. Textbereinigung aller Felder (Leerraum, PDF-Artefakte, Silbentrennungsreste).
2. Zerlegung der Inhalte in Stichpunkte mit Zwischenüberschrift.
3. Entfernung datengetrieben erkannter Standardformulierungen aus den Kompetenztexten.
4. Zerlegung der Literatur in Einzeleinträge mit Abgleichschlüssel `lit_key`.
5. Export nach `data/aufbereitet/` (eine CSV je Tabelle, Format wie `module.csv`).
"""

import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "module.csv"
DEFAULT_OUTPUT = ROOT / "data" / "aufbereitet"

TEXT_FIELDS = [
    "modulname",
    "modulname_en",
    "fachkompetenz",
    "methodenkompetenz",
    "personale_soziale_kompetenz",
    "uebergreifende_handlungskompetenz",
    "lehr_lerneinheiten",
    "inhalte",
    "voraussetzungen",
    "ects",
    "verantwortung",
    "literatur",
]

COMPETENCES = {
    "fachkompetenz": "Fachkompetenz",
    "methodenkompetenz": "Methodenkompetenz",
    "personale_soziale_kompetenz": "Personale und soziale Kompetenz",
    "uebergreifende_handlungskompetenz": "Übergreifende Handlungskompetenz",
}

# ---------------------------------------------------------------------------
# 1. Allgemeine Textbereinigung
# ---------------------------------------------------------------------------

UMLAUT_COMBINING = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}

ARTIFACT_PATTERNS = [
    # Tabellenkopf der Lehr- und Lerneinheiten, der samt Stundenwerten in den Fließtext geraten ist
    re.compile(r"LEHR- UND LERNEINHEITEN PRÄSENZZEIT SELBSTSTUDIUM\s+.+?\s+\d+\s+\d+\s*"),
    re.compile(r"^\s*-?\s*(?:LERNEINHEITEN UND INHALTE|LEHR- UND LERNEINHEITEN)\s*", re.M),
    # Kopf-/Fußzeilen des PDFs (vorsorglich, falls nicht schon bei der Extraktion entfernt)
    re.compile(r"^Stand vom \d{2}\.\d{2}\.\d{4}\s*$", re.M),
    re.compile(r"W3M\d{5} // Seite \d+"),
    re.compile(r"\[/?list\]"),
]

# Wörter nach "Präfix- ", bei denen es sich um eine Ergänzungsstrich-Konstruktion handelt
HYPHEN_KEEP = {
    "und", "oder", "bzw", "sowie", "als", "bis", "u", "vs", "versus", "wie", "beziehungsweise",
    "über", "zu", "and", "or", "noch", "statt", "anstatt", "plus", "&",
}
HYPHEN_RE = re.compile(r"\b([A-Za-zÄÖÜäöüß]{2,})- ([a-zäöüß]{2,})\b")
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def build_vocabulary(rows: list[dict[str, str]]) -> set[str]:
    """Wortschatz (klein geschrieben) über alle Textfelder – Basis für die Silbentrennungsprüfung."""
    vocab: set[str] = set()
    for row in rows:
        for field in TEXT_FIELDS:
            vocab.update(w.lower() for w in WORD_RE.findall(row.get(field, "")))
    return vocab


def dehyphenate(text: str, vocab: set[str], stats: Counter) -> str:
    """Fügt PDF-Trennungen wie "Unter- nehmen" zusammen, lässt "Wirtschafts- und" stehen.

    Zusammengefügt wird nur, wenn das Folgewort keine Konjunktion o. Ä. ist und das zusammen-
    gesetzte Wort im Korpus vorkommt oder das Folgewort allein nicht vorkommt.
    """

    def repl(m: re.Match) -> str:
        first, second = m.group(1), m.group(2)
        if second.lower() in HYPHEN_KEEP:
            return m.group(0)
        joined = first + second
        if joined.lower() in vocab or second.lower() not in vocab:
            stats[f"{first}- {second} → {joined}"] += 1
            return joined
        return m.group(0)

    return HYPHEN_RE.sub(repl, text)


def clean_text(text: str, vocab: set[str], stats: Counter) -> str:
    """Normalisiert Unicode, entfernt PDF-/Tabellenartefakte und vereinheitlicht Leerraum."""
    if not text:
        return ""
    text = re.sub(r"([aouAOU])¨", lambda m: UMLAUT_COMBINING[m.group(1)], text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "").replace("\r", "\n")
    text = re.sub(r"¬\s*", "", text)  # Trennzeichen aus dem PDF ("widerspruchs¬freien")
    text = re.sub(r"[\u00a0\u2007\u2009\u202f\t]", " ", text)
    text = text.replace("[*]", "\n- ")
    for pattern in ARTIFACT_PATTERNS:
        text, n = pattern.subn("\n", text)
        if n:
            stats[f"Artefakt entfernt: {pattern.pattern[:50]}"] += n
    text = dehyphenate(text, vocab, stats)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r" {2,}", " ", line).strip()
        # Aufzählungszeichen vereinheitlichen ("•", "?", "▪", "-Text" am Zeilenanfang)
        line = re.sub(r"^(?:[-–•▪]|\?(?=\s))\s*(?:\?\s+)?", "- ", line) if re.match(
            r"^(?:[-–•▪]|\?\s)", line
        ) else line
        if line in {"-", "- "}:
            continue
        if line:
            lines.append(line)
    return "\n".join(lines)


def flat(text: str) -> str:
    """Einzeilige Fassung (für Embedding-/Vergleichstexte)."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 2. Inhalte → Stichpunkte
# ---------------------------------------------------------------------------

ABBREVIATIONS = {
    "z.b", "b", "u.a", "a", "d.h", "h", "bzw", "ggf", "etc", "ca", "vgl", "i.d.r", "r", "sog",
    "inkl", "evtl", "nr", "bspw", "insb", "u.ä", "o.ä", "ä", "z.t", "t", "e.g", "i.e", "vs", "dr",
    "prof", "st", "bsp", "z. b", "s", "ff", "abs", "art", "sowie", "incl", "zzgl", "max", "min", "mind",
    "v", "e", "u", "o", "i", "d", "z", "jh", "jhd", "ggü", "lt", "gem", "bzgl", "usw", "al", "ua",
}
BULLET_RE = re.compile(r"^- ")
NUMBERED_RE = re.compile(r"^\(?\d{1,2}[.)]\s+\S")
ITEM_SEP_RE = re.compile(r"\s+[-–•]\s+|\s*•\s*|\s+–(?=[A-ZÄÖÜ])|\s\?\s(?=[A-ZÄÖÜ])")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"(])")
INLINE_HEAD_RE = re.compile(r"^([^:()]{2,60}?):\s+(\S.*)$")


def mask_parens(text: str) -> str:
    """Ersetzt Inhalte in Klammern durch Platzhalter gleicher Länge (für Split auf oberster Ebene)."""
    out, depth = [], 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth:
            depth -= 1
            out.append(ch)
            continue
        out.append("\x00" if depth else ch)
    return "".join(out)


def split_top(text: str, pattern: re.Pattern) -> list[str]:
    """Split an `pattern`, aber nicht innerhalb von Klammern."""
    masked = mask_parens(text)
    parts, last = [], 0
    for m in pattern.finditer(masked):
        if m.start() == 0:
            continue
        parts.append(text[last:m.start()])
        last = m.end()
    parts.append(text[last:])
    return [p.strip(" ;,") for p in parts if p.strip(" ;,")]


def split_sentences(text: str) -> list[str]:
    pieces = split_top(text, SENTENCE_RE)
    merged: list[str] = []
    for piece in pieces:
        if merged:
            last_tok = merged[-1].rsplit(" ", 1)[-1].rstrip(".").lower()
            if last_tok in ABBREVIATIONS or re.fullmatch(r"\d+", last_tok):
                merged[-1] += " " + piece
                continue
        merged.append(piece)
    return merged


def split_comma_list(text: str) -> list[str]:
    """Lange, kommagetrennte Stichwortlisten (ohne Satzende) in Einzelstichpunkte zerlegen."""
    parts = split_top(text, re.compile(r",\s+"))
    merged: list[str] = []
    for part in parts:
        if merged and (part[:1].islower() or part[:1] == "-" or re.match(r"(?:z\.\s?B\.|u\.a\.|etc\.|bzw\.)", part)):
            merged[-1] += ", " + part
        else:
            merged.append(part)
    return merged


def is_heading_line(line: str, nxt: str | None) -> bool:
    words = line.split()
    if len(line) <= 80 and line.upper() == line and sum(c.isalpha() for c in line) >= 3:
        return True
    if line.endswith(":") and len(words) <= 10:
        return True
    if line.endswith((".", ",", ";")):
        return False
    if nxt is not None and (BULLET_RE.match(nxt) or NUMBERED_RE.match(nxt)) and len(words) <= 15:
        return True
    return bool(nxt is not None and len(words) <= 5 and "," not in line and len(nxt.split()) > len(words))


def split_items(text: str) -> list[str]:
    """Zerlegt eine Zeile (ohne Aufzählungszeichen) in Einzelstichpunkte."""
    items = split_top(text, ITEM_SEP_RE)
    out: list[str] = []
    for item in items:
        sentences = split_sentences(item)
        if len(sentences) > 1:
            out.extend(sentences)
        elif item.count(", ") >= 3 and len(item) > 120 and not item.endswith("."):
            out.extend(split_comma_list(item))
        else:
            out.append(item)
    return [o.strip(" -–;,•") for o in out if len(o.strip(" -–;,•")) > 1]


TRAILING_HEAD_RE = re.compile(
    r"(?:(?<![\w\-])(?P<adj>[A-ZÄÖÜ][a-zäöüß]+(?:e|es|er|en|em))\s)?"
    r"(?<![\w\-])(?P<head>(?:[A-ZÄÖÜ][\w\-]*-\s(?:und|oder)\s)?[A-ZÄÖÜ][\w\-]*(?:\s\([^)]*\))?)$"
)


CAPS_HEAD_RE = re.compile(r"^((?:[A-ZÄÖÜ0-9&\-,]{2,}\s+){2,}[A-ZÄÖÜ0-9&\-]{2,})\s+(?=[A-ZÄÖÜ][a-zäöüß])(.+)$")


def split_trailing_heading(text: str) -> tuple[str, str]:
    """ "Dokumentation eines IKS Interne Revision" → ("Dokumentation eines IKS", "Interne Revision")."""
    m = TRAILING_HEAD_RE.search(text)
    if not m:
        return text, ""
    start = m.start("adj") if m.group("adj") and text[: m.start("adj")].strip() else m.start("head")
    body = text[:start].strip(" ,;-–")
    return body, text[start:].strip()


def split_contents(inhalte: str) -> tuple[list[tuple[str, str]], str]:
    """Liefert [(zwischenueberschrift, stichpunkt)] und ggf. abgeschnittene BESONDERHEITEN."""
    besonderheiten = ""
    m = re.search(r"^BESONDERHEITEN\s*$", inhalte, re.M)
    if m:
        besonderheiten = inhalte[m.end():].strip()
        inhalte = inhalte[:m.start()]
    lines = [ln for ln in inhalte.split("\n") if ln.strip()]
    top, sub = "", ""
    result: list[tuple[str, str]] = []

    def head() -> str:
        return " > ".join(h for h in (top, sub) if h)

    def add(items: list[str]) -> None:
        """Stichpunkte übernehmen; "… Neue Überschrift:" am Ende eines Stichpunkts wechselt die
        Zwischenüberschrift für die folgenden Stichpunkte (Fließtext-Aufzählungen)."""
        nonlocal top, sub
        for item in items:
            if item.endswith(":") and len(item.split()) > 1:
                body, new_head = split_trailing_heading(item[:-1])
                if body:
                    result.append((head(), body))
                top, sub = new_head, ""
            elif item.endswith(":"):
                top, sub = item[:-1], ""
            else:
                result.append((head(), item))

    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        caps = CAPS_HEAD_RE.match(line)
        if caps:  # "EINFLUSS DER DIGITALISIERUNG … Prozesselemente …"
            top, sub = caps.group(1), ""
            line = caps.group(2)
        if BULLET_RE.match(line):
            add(split_items(line[2:]))
            continue
        if NUMBERED_RE.match(line):
            if nxt is not None and BULLET_RE.match(nxt):
                sub = line.rstrip(":")
            else:
                sub = ""
                add(split_items(line))
            continue
        inline = INLINE_HEAD_RE.match(line)
        if inline and len(inline.group(1).split()) <= 6 and "z.B" not in inline.group(1):
            top, sub = inline.group(1).strip(), ""
            add(split_items(inline.group(2)))
            continue
        parts = split_top(line, ITEM_SEP_RE)
        if is_heading_line(parts[0], nxt if len(parts) == 1 else "- x"):
            top, sub = parts[0].rstrip(":").strip(), ""
            for part in parts[1:]:
                add(split_items(part))
            continue
        add(split_items(line))
    return result, besonderheiten


def contents_text(name: str, points: list[tuple[str, str]]) -> str:
    """Embedding-Text: Modulname + Stichpunkte, gruppiert nach Zwischenüberschrift."""
    blocks: list[str] = [name]
    current, items = None, []
    for heading, item in points + [("\x00", "")]:
        if heading != current:
            if items:
                blocks.append((f"{current}: " if current else "") + "; ".join(items))
            current, items = heading, []
        if item:
            items.append(item)
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# 3. Kompetenzen: datengetriebene Floskelerkennung
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[\wÄÖÜäöüß*\-–]+|[,;:]")
MIN_DF = 8  # Präfix muss in mindestens so vielen Modulen vorkommen
MAX_PREFIX = 12
# Ergänzende Startliste (u. a. englische Module, die zu selten für die Statistik sind)
SEED_PHRASES = [
    "die studierenden sind nach abschluss des moduls in der lage ,",
    "nach abschluss des moduls sind die studierenden in der lage ,",
    "die studierenden sind in der lage ,",
    "die studierenden können",
    "sie sind in der lage ,",
    "sie können",
    "students are able to",
    "students",
    "the students are able to",
    "after completing the module , students are able to",
    "students can",
    "they are able to",
    "they can",
    "they have",
]


def sentences_of(text: str) -> list[str]:
    text = text.replace("…", " ").replace("•", "\n")
    out = []
    for line in text.split("\n"):
        line = line.strip(" -")
        if line:
            out.extend(split_sentences(line))
    return out


def tokens(sentence: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in TOKEN_RE.finditer(sentence)]


def learn_phrases(rows: list[dict[str, str]]) -> dict[tuple[str, ...], int]:
    """Zählt Satzanfangs-n-Gramme (Dokumentfrequenz über Module) in allen Kompetenzfeldern."""
    df: Counter = Counter()
    for row in rows:
        seen: set[tuple[str, ...]] = set()
        for field in COMPETENCES:
            for sentence in sentences_of(row[field]):
                toks = [t for t, _, _ in tokens(sentence)]
                for n in range(2, min(MAX_PREFIX, len(toks) - 1) + 1):
                    seen.add(tuple(toks[:n]))
        df.update(seen)
    function_words = STOP_TITLE | {",", ";", ":"}
    # reine Funktionswort-Präfixe ("durch die") sind keine Floskeln
    phrases = {p: c for p, c in df.items() if c >= MIN_DF and not set(p) <= function_words}
    # Präfixe, die nur auf einem häufigen Wort wie "die" oder einem Inhaltswort enden, werden
    # behalten – der Rest des Satzes bleibt ja erhalten. Kürzere Varianten bleiben ebenfalls.
    for seed in SEED_PHRASES:
        key = tuple(seed.split())
        phrases.setdefault(key, df.get(key, 0))
    phrases.setdefault(("students",), df.get(("students",), 0))
    return phrases


CONJ_START = {"und", "sowie", "oder", "bzw", "and", "or", ","}
SUBJECTS = r"(?:sie|die studierenden|die absolventinnen und absolventen|they|students)"
INVERTED_RE = re.compile(rf"^[\wäöüß]+\s+{SUBJECTS}\b[\s,]*", re.I)  # "haben sie …", "können sie …"


def strip_phrases(
    text: str, phrases: dict[tuple[str, ...], int], used: Counter
) -> str:
    """Entfernt pro Satz den längsten bekannten Floskel-Präfix (wiederholt, max. 3-mal).

    Ein Präfix wird nur entfernt, wenn der Rest nicht mit einer Konjunktion beginnt
    ("Die Studierenden erkennen | und formulieren …" → kürzerer Präfix). Nach einer entfernten
    Satzeinleitung ("Darüber hinaus") wird ein invertiertes Subjekt ("haben sie") mit entfernt.
    """
    kept = []
    for sentence in sentences_of(text):
        toks_all = [t for t, _, _ in tokens(sentence)]
        if tuple(toks_all) in phrases or " ".join(toks_all) in SEED_PHRASES:
            used[tuple(toks_all)] += 1
            continue  # Satz besteht nur aus der Floskel ("Students …")
        for _ in range(3):
            toks = tokens(sentence)
            best = None
            for n in range(min(MAX_PREFIX, len(toks) - 1), 1, -1):
                key = tuple(t for t, _, _ in toks[:n])
                if key in phrases and toks[n][0] not in CONJ_START:
                    best = (key, toks[n - 1][2])
                    break
            if not best:
                break
            used[best[0]] += 1
            sentence = sentence[best[1]:].lstrip(" ,;:")
            inv = INVERTED_RE.match(sentence)
            if inv:
                sentence = sentence[inv.end():]
        sentence = sentence.strip()
        if len(sentence) > 2:
            kept.append(sentence[0].upper() + sentence[1:])
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 4. Literatur
# ---------------------------------------------------------------------------

LIT_FLOSKELN = [
    r"Es wird (?:jeweils )?die (?:jeweils )?(?:aktuellste|aktuelle|neueste) Auflage zu ?Grunde gelegt\.?",
    r"Es gilt jeweils die (?:aktuellste|aktuelle|neueste) Auflage\.?",
    r"The most recent edition is to be used in each case\.?",
    r"(?:Jeweils )?(?:in der )?(?:aktuellste|aktuellen|neuesten) Auflage\.?",
    r"Bitte wie folgt ordnen:.*?\]",
    r"zzgl\. themenspezifische(?:r)? Literatur\.?",
    r"(?:sowie|zusätzlich:?|ergänzend:?)?\s*[{(]?aktuelle[)}]? (?:Artikel|Fachartikel|Veröffentlichungen)\b[^\n]*",
    r"Weitere Unterlagen:[^\n]*",
    r"\b(?:Lehrbücher|Kommentare|Weitere Literatur|Ergänzende Literatur|Vertiefende Literatur|"
    r"Basisliteratur|Grundlagenliteratur|Pflichtliteratur|Literatur|Ergänzend|Zeitschriften|"
    r"Fachwissenschaftliche Journals|Gesetzestexte|Standardwerke)\s*:",
    r"(?:^|(?<=[.\s]))(?:Lehrbücher|Kommentare)(?=\s+[A-ZÄÖÜ])",
]
LIT_FLOSKEL_RE = re.compile("|".join(f"(?:{p})" for p in LIT_FLOSKELN), re.I | re.M)
NOTE_RE = re.compile(
    r"Literatur|Veröffentlichung|Fachzeitschrift|Lehrveranstaltung|bekannt gegeben|Reiseziel|"
    r"Artikel|Journals?\b|Skript|Unterlagen|Hinweis|Abhängigkeit|wird .* zur Verfügung",
    re.I,
)

SEED_PLACES = [
    p.strip()
    for p in """
    Aachen; Amsterdam; Augsburg; Bad Homburg; Bad Soden; Bad Wörishofen; Baden-Baden; Bamberg; Basel;
    Bayreuth; Bergisch Gladbach; Berlin; Bern; Beverly Hills; Bielefeld; Bingley; Bochum; Bonn; Boston;
    Braunschweig; Bremen; Burlington; Cambridge; Cham; Chicago; Chichester; Chippenham; Darmstadt;
    Dortmund; Dresden; Düsseldorf; Edinburgh; Englewood Cliffs; Erlangen; Essen; Frankfurt;
    Frankfurt am Main; Frankfurt/M; Frankfurt a. M; Frankfurt a.M; Frechen; Freiburg; Freiburg im Breisgau; Gießen; Göttingen; Graz; Hallbergmoos;
    Hamburg; Hannover; Harlow; Heidelberg; Herne; Hershey; Hoboken; Ingolstadt; Innsbruck; Jena;
    Karlsruhe; Kassel; Kempten; Kiel; Köln; Konstanz; Landsberg; Landsberg am Lech; Leipzig; Lohmar;
    London; Los Angeles; Ludwigshafen; Mahwah; Mainz; Malden; Mannheim; Marburg; Mason; München;
    Münster; Neuwied; New Delhi; New York; Newbury Park; Norderstedt; Nürnberg; Offenbach; Oldenburg;
    Oxford; Paderborn; Paris; Petersfield; Planegg; Potsdam; Princeton; Regensburg; Rinteln;
    Saarbrücken; San Francisco; Sebastopol; Shanghai; Simmozheim; Singapore; Singapur; Sternenfels;
    Stuttgart; Sydney; Thousand Oaks; Toronto; Tübingen; Ulm; Upper Saddle River; Warschau; Washington;
    Weinheim; Wien; Wiesbaden; Wolfsburg; Würzburg; Zürich
    """.split(";")
    if p.strip()
]
PUBLISHER_RE = re.compile(
    r"\b(?:Verlag|Verl\.|Press|Springer|Gabler|Vahlen|Pearson|Wiley|Schäffer[- ]Poeschel|Haufe|"
    r"Kohlhammer|UTB|UVK|Hanser|Routledge|Sage|McGraw[- ]?Hill|University|De ?Gruyter|Oldenbourg|"
    r"NWB|Erich Schmidt|Beck|dtv|Campus|Penguin|Kogan Page|Palgrave|Macmillan|Norton|Blackwell|"
    r"Thomson|mitp|dpunkt|O[‘’']Reilly|Rheinwerk|Lucius|Böhlau|Uhlenbruch|Haupt|Kiehl|Herder|"
    r"Elsevier|Cengage|Prentice|Addison|Wesley|Vandenhoeck|Ruprecht|Akademikerverlag|VDM|Utb|"
    r"Lexware|Publishing|Publishers|Ltd|Inc|GmbH|AV|Wydawnictwo|Zukunftsinstitut|Kogan|Hrsg)\b",
    re.I,
)

UPPER = "A-ZÀ-ÖØ-ÞŁŚŠŽČ"
LOWER_PLAIN = "a-zß-öø-ÿĀ-ſ"
LOWER = LOWER_PLAIN + "’'´"
PARTICLE = r"(?:(?:[Vv]an [Dd]e[rn]?|[Vv]on [Dd]e[rn]|[Vv]on|[Vv]an|[Dd]e|[Dd]el|[Dd]i|[Ll]e|[Ll]a|ten)\s)?"
SURNAME = (
    rf"{PARTICLE}(?:O[’']|Mc|Mac)?[{UPPER}][{LOWER}]+(?:[-–][{UPPER}][{LOWER}]+)*"
    r"(?:\s(?:Jr\.|III|II)(?![\w]))?"
)
INIT1 = rf"(?:[{UPPER}][a-z]?|[{UPPER}]{{2}})\."
INITS_DOT = rf"{INIT1}(?:\s?-?\s?{INIT1})*"
INITS = rf"(?:{INITS_DOT}|[{UPPER}]-[{UPPER}]\.?|[{UPPER}](?![\w.\-]))"
FIRSTNAMES = rf"[{UPPER}][{LOWER}]+(?:[-\s][{UPPER}][{LOWER}]+){{0,2}}\.?(?:\s{INIT1})*(?:\s(?:von|van|de|zu))?"
P_INIT = rf"{SURNAME},?\s?{INITS}"  # "Baetge, J." / "Kuckartz U." / "Schneider, K"
P_FIRST = rf"{SURNAME},\s{FIRSTNAMES}"  # "Weber, Jürgen"
P_FRONT = rf"{INITS_DOT}\s?{SURNAME}"  # "W. S. Cleveland"
P_VANC = rf"{SURNAME}\s[{UPPER}]{{1,3}}(?=[\s,(])"  # "Lvov E" (Vancouver)
PERSON = rf"(?:{P_INIT}|{P_FIRST}|{P_FRONT}|{P_VANC}|{SURNAME})"
SEP = r"(?:\s*(?:;|/|,\s*&|,\s*und|,|&|\bund\b|\band\b)\s*)"
GROUP = rf"{PERSON}(?:{SEP}{PERSON}){{0,11}}"
HEAD = rf"(?:{PERSON}{SEP}){{0,11}}"
SUFFIX = r"(?P<suffix>\s*,?\s*(?:u\.\s?a\.|et\.?\s?al\.?|u\.\s?v\.\s?m\.))?"
HRSG = r"(?P<hrsg>\.?\s*\((?:Hrsg|Hg|Eds?|eds?|Ed|edts?|Herausgeber)\.*(?:,\s*(?P<hy>\d{4}))?\))?"
YEAR = r"(?P<year>\.?\s*\((?P<y>\d{4}[a-z]?(?:/\d{4})?)\))"
# Je Abschlussart ein Muster; gewählt wird der längste gültige Treffer an einer Position.
AUTHOR_PATTERNS = {
    "colon": re.compile(rf"(?P<group>{GROUP}){SUFFIX}{HRSG}(?:{YEAR})?\.?\s*:(?!//)"),
    "year": re.compile(rf"(?P<group>{GROUP}){SUFFIX}{HRSG}(?:{YEAR})?\s*[.:,;]?"),
    "semi_init": re.compile(rf"(?P<group>{HEAD}{P_INIT}){SUFFIX}{HRSG}\s*[;,]"),
    "semi_first": re.compile(rf"(?P<group>{HEAD}{P_FIRST}){SUFFIX}{HRSG}\s*;"),
    "comma_first": re.compile(rf"(?P<group>(?:{PERSON}{SEP})+{P_FIRST}){HRSG}\s*,"),
    "quote": re.compile(rf"(?P<group>{GROUP}){SUFFIX}{HRSG}(?:{YEAR})?,?\s*(?=[„“”\"])"),
    "front_dot": re.compile(rf"(?P<group>{P_FRONT}(?:{SEP}{P_FRONT}){{0,11}}){SUFFIX}\.\s"),
    "implicit": re.compile(rf"(?P<group>{HEAD}{SURNAME},?\s{INITS_DOT}){SUFFIX}\s(?=[{UPPER}])"),
}
CORP_RE = re.compile(
    rf"(?P<group>[{UPPER}][\w&´’'\-]*(?:\s(?:[{UPPER}][\w&´’'\-.]*|für|der|des|of|and|for|the|und)){{0,7}})"
    r"(?P<hrsg>\s*\((?:Hrsg|Hg|Eds?|[A-Z]{2,6}|\d{4})\.?\))(?:\s*\((?P<y>\d{4})\))?\s*[:.]"
)
NOT_AUTHOR = {"in", "aus", "from", "online", "abruf", "vol", "band", "heft", "teil", "kapitel", "siehe",
              "hrsg", "ed", "eds", "verlag", "isbn", "doi", "available", "url", "stand"}
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def hard_before(before: str) -> bool:
    """Steht vor der Position eine klare Eintragsgrenze?"""
    return not before or before.endswith((".", ";", ")", "?", "!")) or bool(re.search(r"\d$", before))


class AuthorHit:
    """Treffer einer Autorengruppe am Eintragsanfang."""

    def __init__(self, m: re.Match, kind: str):
        self.kind = kind
        self.start, self.end = m.start(), m.end()
        self.text = m.group(0)
        self.group = m.group("group").strip()
        gd = m.groupdict()
        self.year = gd.get("y") or gd.get("hy") or ""
        self.has_hrsg = bool(gd.get("hrsg"))
        self.has_suffix = bool(gd.get("suffix"))
        if kind == "corp":
            self.strong = True
        elif kind in {"colon", "year", "quote"}:
            self.strong = bool(
                re.search(rf"[/;&]|,\s|\s{INIT1}|\s[{UPPER}](?![\w])", self.group)
                or self.has_hrsg or self.has_suffix or (self.year and kind == "colon")
            )
        else:
            self.strong = True
        if kind == "comma_first" and not re.search(r"[/;&]", self.group) or kind == "year" and not (self.year or self.has_hrsg):
            self.valid = False
        else:
            self.valid = self.group.split()[0].strip(",.:;").lower() not in NOT_AUTHOR

    @property
    def first_surname(self) -> str:
        group = self.group
        if re.match(rf"{INITS_DOT}\s?[{UPPER}]", group):  # Initialen vorangestellt
            group = re.sub(rf"^{INITS_DOT}\s?", "", group)
        first = re.split(r"\s*(?:,|;|/|&|\bund\b|\band\b)\s*", group)[0]
        first = re.sub(rf"\s(?:{INITS}|[{UPPER}]{{1,3}})$", "", first)  # "Kuckartz U." / "Lvov E"
        return first.strip()
STOP_TITLE = set(
    """der die das des dem den ein eine einer eines einem einen und oder für von zu zur zum im in
    mit auf aus bei nach über unter vom am an als wie the a an of and for to in on with from by
    at its their your how what is are be le la les et des du""".split()
)


def ascii_fold(text: str) -> str:
    text = text.replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def norm_token(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", ascii_fold(word).lower())


NAME_PREFIX_RE = re.compile(
    rf"(?:{SURNAME})(?<!ment)(?<!ung)(?<!heit)(?<!keit)(?<!schaft)(?<!tion)(?<!ik)(?<!ismus)(?<!lehre)(?<!wesen)"
)


class LitParser:
    """Heuristische Zerlegung und Feldbestimmung für Literaturangaben."""

    def __init__(self, places: set[str]):
        self.set_places(places)

    def set_places(self, places: set[str]) -> None:
        self.places = {p for p in places if p}
        alts = "|".join(re.escape(p) for p in sorted(self.places, key=len, reverse=True))
        place = rf"(?:{alts})"
        self.place_re = re.compile(
            rf"(?<![\w\-])(?P<ort>{place}(?:\s*(?:/|,|und|\s)\s*{place}){{0,3}}"
            rf"(?:,\s[A-Z]{{2}}\b)?(?:\s+(?:u\.\s?a\.|et\.?\s?al\.?|am Main))?)(?![\w\-])"
        )
        self.place_word_re = re.compile(rf"^{place}$")

    # -- Zerlegung --------------------------------------------------------
    def preclean(self, text: str, removed: Counter) -> str:
        """Floskeln entfernen und typische Extraktionsfehler (fehlende Leerzeichen) reparieren."""
        text = re.sub(rf"(?<=[{LOWER_PLAIN}]{{3}})(?=[{UPPER}][{LOWER}]+,\s?[{UPPER}]\.)", " ", text)
        text = re.sub(rf"(?<=\S)\.+(?=[{UPPER}][{LOWER}]+,\s?[{UPPER}])", ". ", text)
        text = re.sub(rf"(?<=[{LOWER_PLAIN}]{{3}})\.(?=[{UPPER}][{LOWER}])", ". ", text)
        text = re.sub(rf"\s\?\s(?=[{UPPER}])", "\n", text)
        text = re.sub(rf"\s-(?=[{UPPER}][{LOWER}]+,)", " - ", text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"(?m)^[-•]\s*(?:\?\s*)?", "", text)
        text = text.replace("&amp;", "&")

        def drop(m: re.Match) -> str:
            removed[flat(m.group(0))[:120]] += 1
            return "\n"

        return LIT_FLOSKEL_RE.sub(drop, text)

    def author_at(self, chunk: str, pos: int) -> AuthorHit | None:
        hits = []
        for kind, pattern in AUTHOR_PATTERNS.items():
            if kind == "implicit" and chunk[:pos].strip(" -–•"):
                continue  # nur am Zeilen-/Eintragsanfang
            m = pattern.match(chunk, pos)
            if m:
                hit = AuthorHit(m, kind)
                if hit.valid and self.plausible(chunk, hit):
                    hits.append(hit)
        if not hits:
            m = CORP_RE.match(chunk, pos)
            return AuthorHit(m, "corp") if m else None
        # längster Treffer; bei Gleichstand der "stärkere"
        return max(hits, key=lambda h: (h.end, h.strong))

    def candidates(self, chunk: str) -> list[AuthorHit]:
        """Alle akzeptierten Eintragsanfänge einer Zeile."""
        starts = [
            m.start()
            for m in re.finditer(
                rf"(?<![\w’'´.\-/])(?:[{UPPER}]|(?:[Vv]on|[Vv]an|[Dd]e|ten)\s[{UPPER}])", chunk
            )
        ]
        accepted: list[AuthorHit] = []
        min_pos = 0
        for pos in starts:
            if pos < min_pos:
                continue
            hit = self.author_at(chunk, pos)
            if hit is None or not self._accept(chunk, pos, hit, accepted):
                continue
            accepted.append(hit)
            min_pos = hit.end
        return accepted

    def _is_place(self, text: str) -> bool:
        return bool(self.place_word_re.match(text.strip(" .,;:()")))

    def plausible(self, chunk: str, hit: AuthorHit) -> bool:
        """Verwirft Treffer, die eher "Titel, Ort: Verlag" als "Autor: Titel" sind."""
        names = re.split(r"\s*(?:,|;|/|&|\bund\b|\band\b)\s*", hit.group)
        if any(self._is_place(n) or PUBLISHER_RE.search(n) for n in names[1:] if n):
            return False  # "Geldtheorie, München:" – Ort/Verlag als vermeintlicher Vorname
        if hit.text.rstrip().endswith(":"):
            seg = re.split(r"[.,;\n(]", chunk[hit.end:], maxsplit=1)[0].strip()
            words = seg.split()
            if len(words) <= 3 and PUBLISHER_RE.search(seg):
                return False  # "München: Vahlen."
            if self.place_re.fullmatch(chunk[hit.end:].strip(" .")):
                return False  # "…, Gabler Verlag: Wiesbaden."
            if not hit.strong and re.fullmatch(
                r"\s*[^\s,.;]+(?:\s[^\s,.;]+)?\s*(?:[,.]\s*(?:19|20)\d\d)?\.?\s*", chunk[hit.end:]
            ):
                return False  # "Auckland: Bantam, 2007"
        return True

    def _accept(self, chunk: str, pos: int, hit: AuthorHit, accepted: list[AuthorHit]) -> bool:
        before = chunk[:pos].rstrip(" -–•")
        prev_tok = before.rsplit(None, 1)[-1] if before else ""
        if prev_tok.lower().rstrip(":") in {"in", "aus", "from", "and", "und", "&", "von", "of", "by", "hrsg."}:
            return False
        if accepted and not re.search(r"\w", chunk[accepted[-1].end:pos]):
            return False  # Titel des vorherigen Eintrags darf nicht leer sein
        if not self.plausible(chunk, hit):
            return False
        if before.endswith(":") and (accepted or len(before.split()) > 4):
            return False  # direkt nach "Titel:" beginnt kein neuer Eintrag
        if not hit.strong and PUBLISHER_RE.fullmatch(hit.first_surname):
            return False
        if (not hard_before(before) and re.fullmatch(r"[a-zäöüß][\w\-]*,?", prev_tok)
                and not PUBLISHER_RE.fullmatch(prev_tok.rstrip(","))):
            return False  # Kleingeschriebenes Wort davor → Kandidat liegt mitten im Titel
        first = hit.first_surname
        if hit.kind == "corp":
            return (not before or before.endswith((".", ";", ")"))) and not self._is_place(hit.group)
        if self._is_place(first) and not re.match(rf"{SURNAME},?\s?{INITS}(?:$|[\s;,/&(:])", hit.group + ":"):
            return False  # "Wiesbaden, Verlag, …" – aber "Bamberg, G." ist ein Autor
        if hit.strong:
            return True
        # schwacher Kandidat ("Kuhn:") – nur am Zeilenanfang oder direkt nach Ort/Verlag/Jahr
        after = chunk[hit.end:]
        if not after.strip() or self._is_place(first) or PUBLISHER_RE.search(hit.group):
            return False
        if self.place_re.match(after.strip()):
            return False  # "Springer: Wiesbaden"
        if not before:
            return True
        prev_clean = prev_tok.strip(".,;:()")
        return bool(
            self._is_place(prev_clean)
            or YEAR_RE.fullmatch(prev_clean)
            or PUBLISHER_RE.fullmatch(prev_clean)
            or prev_tok in {"u.a.", "al.", "u. a.", "o.O.", "o.S.", "o. O.", "o. S."}
        )

    UNFINISHED_RE = re.compile(
        rf"(?:[,/;&:„]|\b(?:und|and|von|van|de|den|der|die|das|dem|ein|eine|einer|ten|des|the|of|for|für|"
        rf"zur|zum|im|in|mit|a|an|zu|vom|beim|on|to|bei|über|unter|als|wie))\s*$|[,/;&]\s*[{UPPER}]\.?\s*$|/\s*[{UPPER}][\w\-]+(?:\s[{UPPER}][\w\-]+)?\s*$"
    )

    def _join_lines(self, lines: list[str]) -> list[str]:
        """Fügt über Zeilenumbrüche zerrissene Einträge wieder zusammen."""
        out: list[str] = []
        for line in lines:
            if out:
                prev = out[-1]
                cont = (
                    self.UNFINISHED_RE.search(prev)
                    or re.fullmatch(rf"{SURNAME},\s{FIRSTNAMES}", prev)
                    or re.search(rf",\s[{UPPER}][{LOWER}]+$", prev) and not self._is_place(prev.rsplit(None, 1)[-1])
                    or line[:1].islower()
                    or re.match(r"[Ii]n:\s", line)
                )
                if not cont and re.search(rf"[{LOWER_PLAIN}]{{3,}}$", prev) and not NOTE_RE.search(prev):
                    hit = self.author_at(line, 0)
                    cont = hit is None or not hit.strong or not self.plausible(line, hit)
                    cont = cont and not (line.endswith(":") or line.startswith("("))
                if cont:
                    out[-1] = f"{prev} {line}"
                    continue
            out.append(line)
        return out

    def split(self, text: str, removed: Counter) -> list[dict]:
        text = self.preclean(text, removed)
        lines = [c.strip(" -–") for c in text.split("\n")]
        lines = self._join_lines([c for c in lines if c and re.search(r"\w", c)])
        entries: list[dict] = []
        for chunk in lines:
            cands = self.candidates(chunk)
            pre = chunk[: cands[0].start] if cands else chunk
            pre = pre.strip(" -–.;•")
            first_start = cands[0].start if cands else 0
            if pre and cands and NAME_PREFIX_RE.fullmatch(pre):
                # "Andelfinger Volker P. und …" – Nachname vor dem erkannten Autorenanfang
                first_start, pre = chunk.find(pre), ""
            if pre:
                self._handle_pretext(pre, entries, removed)
            for i, hit in enumerate(cands):
                start = first_start if i == 0 else hit.start
                end = cands[i + 1].start if i + 1 < len(cands) else len(chunk)
                raw = chunk[start:end].strip(" -–;•")
                offset = len(chunk[start:end]) - len(chunk[start:end].lstrip(" -–;•"))
                entries.append({"raw": raw, "hit": hit, "a_len": hit.end - start - offset,
                                "prefix": chunk[start:hit.start].strip()})
        return entries

    def _handle_pretext(self, pre: str, entries: list[dict], removed: Counter) -> None:
        words = pre.split()
        looks_like_entry = bool(YEAR_RE.search(pre) or self.place_re.search(pre)) and len(words) > 4
        if pre.startswith("(") and pre.endswith(")") or NOTE_RE.search(pre) and not looks_like_entry:
            removed[f"[Hinweis] {pre[:100]}"] += 1
        elif pre.endswith(":") and len(words) <= 8 or re.match(r"TEIL [IVX]+:", pre):
            removed[f"[Zwischenüberschrift] {pre[:100]}"] += 1
        elif entries and (pre[:1].islower() or (len(words) <= 6 and (self.place_re.search(pre) or YEAR_RE.search(pre)))):
            entries[-1]["raw"] += " " + pre  # Fortsetzung des vorherigen Eintrags
        elif len(words) <= 8 and "," not in pre and not YEAR_RE.search(pre):
            removed[f"[Zwischenüberschrift] {pre[:100]}"] += 1
        else:
            entries.append({"raw": pre, "hit": None, "a_len": 0})

    # -- Felder -----------------------------------------------------------
    def fields(self, entry: dict) -> dict:
        raw = flat(entry["raw"])
        hit: AuthorHit | None = entry["hit"]
        reasons: list[str] = []
        if hit is None:
            authors, first, rest, year = "", "", raw, ""
            reasons.append("kein Autor erkannt")
        else:
            authors = flat(raw[: entry["a_len"]]).rstrip(":;,. ")
            first = entry.get("prefix") or hit.first_surname
            rest = raw[entry["a_len"]:].strip()
            year = hit.year
            if not hit.strong:
                reasons.append("Autor ohne Initialen/Trenner")
        year_m = YEAR_RE.search(year) or YEAR_RE.search(rest)
        rest = re.sub(r"\(Paperback\)|\(?ISBN[^,)]*\)?|Online unter\s+\S+|https?://\S+|www\.\S+", "", rest)
        title, ort = self._title_place(rest, apa=bool(hit and (hit.year or hit.kind == "year")))
        if not title:
            reasons.append("kein Titel")
        if len(title.split()) > 25:
            reasons.append("Titel sehr lang (evtl. verschmolzene Einträge)")
        if re.search(rf"[{UPPER}][{LOWER}]+,\s?[{UPPER}]\.\s?[:;,]", title):
            reasons.append("Autorenmuster im Titel")
        if not ort and len(title.split()) > 15:
            reasons.append("kein Ort, langer Titel")
        return {
            "eintrag_roh": raw,
            "autoren_roh": authors,
            "erstautor_nachname": first,
            "titel": title,
            "ort": ort,
            "jahr": year_m.group(0) if year_m else "",
            "lit_key": self.key(first, title),
            "parse_unsicher": bool(reasons),
            "unsicher_grund": "; ".join(reasons),
        }

    def _title_place(self, rest: str, apa: bool) -> tuple[str, str]:
        rest = rest.strip(" .,;:")
        ort = ""
        quoted = re.match(r"[„“”\"]\s*(.+?)\s*[“”\"]", rest)
        if quoted:
            # Titel in Anführungszeichen: Titel = Zitat, Ort aus dem Rest
            tail = rest[quoted.end():]
            places = list(self.place_re.finditer(tail))
            return flat(quoted.group(1)).strip(" ,.;:"), places[-1].group("ort") if places else ""
        places = [m for m in self.place_re.finditer(rest) if m.start() > 3]
        cut = len(rest)
        if places:
            last = places[-1]
            ort = re.sub(r",\s[A-Z]{2}$|\s+(?:u\.\s?a\.|et\.?\s?al\.?)$", "", last.group("ort")).strip()
            cut = places[0].start() if apa else last.start()
        edition = re.search(
            r"[,.]?\s*(?:\(?\d+\.?\s?(?:st|nd|rd|th)?\.? ?ed(?:ition)?\.?\)?|\d+\.,?\s*(?:vollst|überarb|aktual|"
            r"erw|unveränd|neu|korr|durchges|Aufl)|\bAufl(?:age)?\b|\b(?:First|Second|Third|Fourth|Fifth|Sixth|"
            r"Seventh|Eighth|Ninth|Tenth|Revised|International)\s+ed(?:ition)?\b)",
            rest,
            re.I,
        )
        if edition and edition.start() > 3:
            cut = min(cut, edition.start())
        inref = re.search(r"[,.;]?\s+[Ii]n:\s", rest)
        if inref and inref.start() > 3:
            cut = min(cut, inref.start())
        if not ort:
            # "…, Verlag, Ort" mit unbekanntem Ort (z. B. "Rainer Hampp Verlag, Mering")
            m = re.search(rf",\s*[^,]*{PUBLISHER_RE.pattern}[^,]*,\s*([{UPPER}][\w\-]+(?:[ /][{UPPER}][\w\-]+)?)\.?$",
                          rest, re.I)
            if m and m.start() > 3:
                ort, cut = m.group(1), min(cut, m.start())
        title = rest[:cut]
        if apa:
            # APA: Titel endet am ersten Satzende
            first_sentence = re.split(r"(?<=[\w)\]])[.?]\s+(?=[A-ZÄÖÜ])", title, maxsplit=1)[0]
            title = first_sentence
        # Verlags-, Jahres- und Ortsreste am Ende abschneiden
        for _ in range(4):
            title = title.strip(" .,;:/")
            parts = re.split(r",\s+|\.\s+|:\s+", title)
            if len(parts) > 1 and (
                PUBLISHER_RE.search(parts[-1]) and len(parts[-1].split()) <= 4
                or YEAR_RE.fullmatch(parts[-1].strip("() "))
                or re.fullmatch(r"(?:S\.\s?)?\d+\s?[-–]\s?\d+|Bd\.\s?\d+|Band \d+|o\.\s?[SO]\.", parts[-1].strip())
            ):
                title = title[: title.rfind(parts[-1])]
                continue
            break
        title = re.sub(r"\s*\(?\b(?:19|20)\d{2}\)?$", "", title).strip(" .,;:/–-")
        title = re.sub(r",\s*(?:[A-Z]\.){1,3}[A-Z]?$", "", title)  # Rest wie "C.H" (C.H. Beck)
        if title.count("(") > title.count(")"):
            title = title[: title.rfind("(")].rstrip(" ,.;:")
        return flat(title), ort

    @staticmethod
    def key(first: str, title: str) -> str:
        """Erstautor (ohne Diakritika, klein) + erste 3 bedeutungstragende Wörter des Haupttitels.

        Der Haupttitel endet am ersten Untertitel-Trenner (":", " – ", " - ", ". "), damit
        Angaben mit und ohne Untertitel denselben Schlüssel erhalten.
        """
        parts = first.split()
        surname = parts[-1] if len(parts) > 1 and parts[0].lower() in {"von", "van", "de"} else first
        main = re.split(r":\s|\s[–-]\s|\.\s|\?\s", title, maxsplit=1)[0]
        words = [norm_token(w) for w in re.split(r"[\s\-–/:,.;()“”„\"'&+]+", main)]
        meaningful = [w for w in words if w and w not in STOP_TITLE][:3]
        if not surname and not meaningful:
            return ""
        return "_".join([norm_token(surname) or "ohneautor"] + meaningful)


PLACE_STOP = {"Auflage", "Aufl", "York", "Edition", "Band", "Verlag", "Vol", "Heft", "Teil", "Online", "Hrsg"}


def learn_places(parser: LitParser, rows: list[dict[str, str]]) -> set[str]:
    """Ergänzt die Ortsliste datengetrieben.

    Quellen: (a) "…, Verlag, Ort." am Eintragsende, (b) "Ort: Verlag" und (c) Orte am Ende klar
    getrennter Zeilen. Übernommen wird, was mindestens zweimal (a/b) bzw. dreimal (c) vorkommt.
    """
    word = rf"[{UPPER}][a-zäöüß]+(?:[ -][{UPPER}][a-zäöüß]+)?"
    strong_counts: Counter = Counter()
    weak_counts: Counter = Counter()
    pub_place = re.compile(rf",\s*[^,]*{PUBLISHER_RE.pattern}[^,]*,\s*({word})\.?(?=$|[,\s]+(?:19|20)\d\d)", re.I)
    place_pub = re.compile(rf"(?:^|[,.]\s)({word}):\s*[^.,:]{{0,30}}{PUBLISHER_RE.pattern}", re.I)
    end_re = re.compile(
        rf"(?:,|:|\.)\s*({word}(?:/{word})?)(?:\s+(?:u\.\s?a\.|et al\.))?(?:,?\s*(?:19|20)\d{{2}})?\.?$"
    )
    for row in rows:
        for line in row["literatur"].split("\n"):
            line = line.strip(" -")
            for m in pub_place.finditer(line):
                strong_counts[m.group(1)] += 1
            for m in place_pub.finditer(line):
                strong_counts[m.group(1)] += 1
            m = end_re.search(line)
            if m:
                for part in m.group(1).split("/"):
                    weak_counts[part] += 1
    known = set(parser.places)

    def ok(w: str) -> bool:
        return (
            w not in known and w not in PLACE_STOP and not PUBLISHER_RE.search(w)
            and w.lower() not in STOP_TITLE and len(w) > 3
            and not re.search(r"(?:ung|keit|heit|ment|ing|tion|schaft|ismus|lehre|recht|ik|ie)$", w)
        )

    learned = {w for w, c in strong_counts.items() if c >= 2 and ok(w)}
    learned |= {w for w, c in weak_counts.items() if c >= 3 and ok(w)}
    return known | learned


# ---------------------------------------------------------------------------
# 5. CSV-Export
# ---------------------------------------------------------------------------


def write_csvs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Eine CSV je Tabelle, im Format der Original-`module.csv` (UTF-8 mit BOM, alle Felder quotiert)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL,
                  lineterminator="\r\n")


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def preprocess(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict:
    raw_rows = read_rows(input_path)
    vocab = build_vocabulary(raw_rows)
    clean_stats: Counter = Counter()
    rows = []
    for raw in raw_rows:
        row = dict(raw)
        for field in TEXT_FIELDS:
            row[field] = clean_text(raw.get(field, ""), vocab, clean_stats)
        rows.append(row)

    # Kompetenzen
    phrases = learn_phrases(rows)
    used_phrases: Counter = Counter()
    komp_rows = []
    for raw, row in zip(raw_rows, rows):
        kern_parts = []
        for field, label in COMPETENCES.items():
            kern = strip_phrases(row[field], phrases, used_phrases) if row[field] else ""
            row[f"{field}_kern"] = kern
            if kern:
                kern_parts.append(kern)
            komp_rows.append({
                "modulnummer": row["modulnummer"],
                "kompetenzart": label,
                "text_original": raw[field],
                "text_bereinigt": kern,
            })
        row["kompetenzen_kern"] = "\n".join(kern_parts)

    # Inhalte
    point_rows = []
    for row in rows:
        points, besonderheiten = split_contents(row["inhalte"])
        row["_points"] = points
        row["besonderheiten_aus_inhalte"] = besonderheiten
        row["inhalte_bereinigt"] = row["inhalte"].split("\nBESONDERHEITEN")[0].strip()
        for pos, (heading, item) in enumerate(points, start=1):
            point_rows.append({
                "modulnummer": row["modulnummer"],
                "modulname": row["modulname"],
                "position": pos,
                "zwischenueberschrift": heading,
                "stichpunkt": item,
            })

    # Literatur (zweistufig: Ortsliste aus den Daten ergänzen, dann endgültig zerlegen)
    parser = LitParser(set(SEED_PLACES))
    parser.set_places(learn_places(parser, rows))
    lit_removed: Counter = Counter()
    lit_rows = []
    for row in rows:
        entries = parser.split(row["literatur"], lit_removed) if row["literatur"] else []
        for pos, entry in enumerate(entries, start=1):
            lit_rows.append({"modulnummer": row["modulnummer"], "position": pos, **parser.fields(entry)})
        row["_lit"] = len(entries)

    df_points = pd.DataFrame(point_rows)
    df_lit = pd.DataFrame(lit_rows)
    df_komp = pd.DataFrame(komp_rows)

    # Modulblatt
    module_rows = []
    lit_unsure = df_lit.groupby("modulnummer")["parse_unsicher"].sum().to_dict()
    for row in rows:
        module_rows.append({
            "modulnummer": row["modulnummer"],
            "modulname": row["modulname"],
            "modulname_en": row["modulname_en"],
            "ects": row["ects"],
            "verantwortung": row["verantwortung"],
            "lehr_lerneinheiten": row["lehr_lerneinheiten"],
            "voraussetzungen": row["voraussetzungen"],
            "inhalte_bereinigt": row["inhalte_bereinigt"],
            **{f: row[f] for f in COMPETENCES},
            **{f"{f}_kern": row[f"{f}_kern"] for f in COMPETENCES},
            "kompetenzen_kern": row["kompetenzen_kern"],
            "literatur_bereinigt": row["literatur"],
            "text_inhalte": contents_text(row["modulname"], row["_points"]),
            "text_kompetenzen": flat(row["kompetenzen_kern"]),
            "anzahl_stichpunkte": len(row["_points"]),
            "anzahl_literatureintraege": row["_lit"],
            "anzahl_literatur_unsicher": int(lit_unsure.get(row["modulnummer"], 0)),
        })
    df_mod = pd.DataFrame(module_rows)

    # Literatur-Häufigkeit (nur Schlüssel mit Erstautor)
    keyed = df_lit[(df_lit["lit_key"] != "") & ~df_lit["lit_key"].str.startswith("ohneautor")]
    freq = []
    for key, grp in keyed.groupby("lit_key"):
        mods = sorted(set(grp["modulnummer"]), reverse=True)
        freq.append({
            "lit_key": key,
            "anzahl_module": len(mods),
            "modulnummern": "; ".join(mods),
            "beispiel_eintrag": grp["eintrag_roh"].iloc[0],
        })
    df_freq = pd.DataFrame(freq).sort_values(["anzahl_module", "lit_key"], ascending=[False, True])

    # Entfernte Floskeln
    floskel_rows = []
    for key, n in sorted(used_phrases.items(), key=lambda kv: (-kv[1], kv[0])):
        floskel_rows.append({
            "bereich": "Kompetenzen",
            "floskel": " ".join(key).replace(" ,", ",").replace(" ;", ";").replace(" :", ":"),
            "module_mit_satzanfang": phrases.get(key, 0),
            "entfernt_anzahl": n,
            "quelle": "Startliste" if " ".join(key) in SEED_PHRASES and phrases.get(key, 0) < MIN_DF
            else "datengetrieben (n-Gramm-Präfix)",
        })
    for text, n in sorted(lit_removed.items(), key=lambda kv: (-kv[1], kv[0])):
        floskel_rows.append({
            "bereich": "Literatur",
            "floskel": text,
            "module_mit_satzanfang": "",
            "entfernt_anzahl": n,
            "quelle": "Regel (Einleitung/Hinweis/Zwischenüberschrift)",
        })
    for text, n in sorted(clean_stats.items()):
        floskel_rows.append({
            "bereich": "Bereinigung",
            "floskel": text,
            "module_mit_satzanfang": "",
            "entfernt_anzahl": n,
            "quelle": "Artefakt/Silbentrennung",
        })
    df_floskel = pd.DataFrame(floskel_rows)

    df_quality = quality_sheet(raw_rows, df_mod, df_points, df_lit, df_freq)

    tables = {
        "module": df_mod,
        "inhalte_stichpunkte": df_points,
        "kompetenzen": df_komp,
        "literatur": df_lit,
        "literatur_haeufigkeit": df_freq,
        "entfernte_floskeln": df_floskel,
        "qualitaet": df_quality,
    }
    write_csvs(tables, output_path)
    return {
        "module": len(df_mod),
        "stichpunkte": len(df_points),
        "literatur": len(df_lit),
        "unsicher": int(df_lit["parse_unsicher"].sum()),
        "top": df_freq.head(10)[["lit_key", "anzahl_module"]].values.tolist(),
    }


def quality_sheet(raw_rows, df_mod, df_points, df_lit, df_freq) -> pd.DataFrame:
    q: list[dict] = []

    def add(bereich, befund, modul="", wert="", details=""):
        q.append({"bereich": bereich, "befund": befund, "modulnummer": modul, "wert": wert, "details": details})

    n_lit = len(df_lit)
    n_uns = int(df_lit["parse_unsicher"].sum())
    add("Kennzahl", "Module", wert=len(df_mod))
    add("Kennzahl", "Stichpunkte gesamt", wert=len(df_points))
    add("Kennzahl", "Stichpunkte je Modul (Median)", wert=float(df_mod["anzahl_stichpunkte"].median()))
    add("Kennzahl", "Stichpunkte mit Zwischenüberschrift", wert=int((df_points["zwischenueberschrift"] != "").sum()))
    add("Kennzahl", "Literatureinträge gesamt", wert=n_lit)
    add("Kennzahl", "Literatureinträge unsicher", wert=n_uns, details=f"{n_uns / max(n_lit, 1):.1%}")
    add("Kennzahl", "Literatureinträge ohne Ort", wert=int((df_lit["ort"] == "").sum()))
    add("Kennzahl", "Eindeutige lit_keys", wert=int(df_lit["lit_key"].nunique()))
    add("Kennzahl", "lit_keys in ≥2 Modulen", wert=int((df_freq["anzahl_module"] >= 2).sum()))
    for grund, n in Counter(
        g for s in df_lit["unsicher_grund"] for g in s.split("; ") if g
    ).most_common():
        add("Kennzahl", f"Unsicherheitsgrund: {grund}", wert=n)
    for field in TEXT_FIELDS:
        empty = [r["modulnummer"] for r in raw_rows if not r.get(field, "").strip()]
        add("Leere Felder", f"leer: {field}", wert=len(empty), details=", ".join(empty))
    for _, r in df_mod[df_mod["anzahl_literatureintraege"] == 0].iterrows():
        add("Literatur", "Modul ohne Literatureinträge", r["modulnummer"], 0, r["literatur_bereinigt"][:200])
    for _, r in df_mod[df_mod["anzahl_stichpunkte"] <= 1].iterrows():
        add("Inhalte", "Modul mit ≤1 Stichpunkt", r["modulnummer"], r["anzahl_stichpunkte"], r["inhalte_bereinigt"][:200])
    for _, r in df_mod[(df_mod["kompetenzen_kern"] == "")].iterrows():
        add("Kompetenzen", "Kompetenzkern leer", r["modulnummer"])
    for _, r in df_mod.iterrows():
        texts = [(f, r[f]) for f in COMPETENCES if r[f]]
        for i, (f1, t1) in enumerate(texts):
            for f2, t2 in texts[i + 1:]:
                if t1 == t2:
                    add("Kompetenzen", "identischer Text in zwei Kompetenzarten", r["modulnummer"], "", f"{f1} = {f2}")
    for field in COMPETENCES:
        for _, r in df_mod[(df_mod[field] != "") & (df_mod[f"{field}_kern"] == "")].iterrows():
            add("Kompetenzen", f"nach Floskelentfernung leer: {field}", r["modulnummer"], "", r[field][:200])
    lens = df_points["stichpunkt"].str.len()
    for _, r in df_points[lens < 4].iterrows():
        add("Inhalte", "sehr kurzer Stichpunkt (<4 Zeichen)", r["modulnummer"], len(r["stichpunkt"]), r["stichpunkt"])
    for _, r in df_points[lens > 300].iterrows():
        add("Inhalte", "sehr langer Stichpunkt (>300 Zeichen)", r["modulnummer"], len(r["stichpunkt"]), r["stichpunkt"][:300])
    for _, r in df_lit[df_lit["parse_unsicher"]].iterrows():
        add("Literatur", f"unsicherer Parse: {r['unsicher_grund']}", r["modulnummer"], r["position"], r["eintrag_roh"][:300])
    dup = df_lit[df_lit["lit_key"] != ""].duplicated(["modulnummer", "lit_key"], keep="first")
    for _, r in df_lit[df_lit["lit_key"] != ""][dup].iterrows():
        add("Literatur", "Eintrag im Modul doppelt (gleicher lit_key)", r["modulnummer"], r["position"], r["lit_key"])
    for _, r in df_mod[df_mod["modulname"].isin(df_mod["modulname"][df_mod["modulname"].duplicated()])].iterrows():
        add("Module", "Modulname mehrfach vergeben", r["modulnummer"], "", r["modulname"])
    return pd.DataFrame(q)


def main() -> None:
    stats = preprocess()
    share = stats["unsicher"] / max(stats["literatur"], 1)
    print(f"{stats['module']} Module, {stats['stichpunkte']} Stichpunkte, {stats['literatur']} "
          f"Literatureinträge ({stats['unsicher']} unsicher = {share:.1%}) → {DEFAULT_OUTPUT}")
    print("Top-10 geteilte Literatur-Keys:")
    for key, n in stats["top"]:
        print(f"  {n:3d}  {key}")


if __name__ == "__main__":
    main()
