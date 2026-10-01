"""Extrahiert die Module aus dem Markdown-Modulhandbuch in eine CSV (eine Zeile pro Modul)."""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "M_W_Modulhandbuch.md"
DEFAULT_OUTPUT = ROOT / "data" / "module.csv"

MODULE_RE = re.compile(r"^## \*\*(.+?) \((W3M\d{5})\)(?: (.+?))?\s*\*\*\s*$", re.M)
HEADING_RE = re.compile(r"^#{1,6} \*\*(.+?)\*\*\s*$", re.M)

FIELDS = [
    "modulnummer",
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
    "text_gesamt",
]

COMPETENCES = {
    "FACHKOMPETENZ": "fachkompetenz",
    "METHODENKOMPETENZ": "methodenkompetenz",
    "PERSONALE UND SOZIALE KOMPETENZ": "personale_soziale_kompetenz",
    "ÜBERGREIFENDE HANDLUNGSKOMPETENZ": "uebergreifende_handlungskompetenz",
}

# Seitenumbruch-Störtext, der mitten in Abschnitten auftaucht
NOISE_PATTERNS = [
    re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S),
    re.compile(r"^Stand vom \d{2}\.\d{2}\.\d{4}\s*$", re.M),
    re.compile(r"^\*\*W3M\d{5} // Seite \d+\*\*\s*$", re.M),
    re.compile(r"^\*\*AUS AKTUELLER ORGA-EINHEIT\*\*\s*$", re.M),
    re.compile(r"^\*\*(PRÄSENZZEIT|SELBSTSTUDIUM)( SELBSTSTUDIUM)?\*\*\s*$", re.M),
    re.compile(r"^\d+\s*$", re.M),  # lose Stundenzahlen aus den Seitenköpfen
    re.compile(r"^Es wird jeweils die aktuellste Auflage zu Grunde gelegt\.?\s*$", re.M),
]


def clean(text: str) -> str:
    """Störtext entfernen, Markdown-Fettdruck/Bullets glätten, Leerraum normalisieren."""
    for pattern in NOISE_PATTERNS:
        text = pattern.sub("", text)
    lines = []
    for line in text.splitlines():
        line = line.replace("<br>", " ").replace("**", "")
        line = re.sub(r"^\s*[-•]\s+", "- ", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line.startswith("#"):
            line = line.lstrip("# ").strip()
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{2,}", "\n", text).strip()


FLOW_UNITS_RE = re.compile(
    r"^\*\*LERNEINHEITEN UND INHALTE LEHR- UND LERNEINHEITEN PRÄSENZZEIT SELBSTSTUDIUM\*\*\s+"
    r"(.+?)\s+(\d+)\s+(\d+)(?:\s+(.*))?$",
    re.M,
)
SPLIT_UNITS_RE = re.compile(
    r"^\*\*LEHR- UND LERNEINHEITEN\*\*\s+(.+?)\s*\n+(?:\*\*PRÄSENZZEIT SELBSTSTUDIUM\*\*\s+)?(\d+)\s+(\d+)\s*$",
    re.M,
)
FLOW_ECTS_RE = re.compile(
    r"ECTS-LEISTUNGSPUNKTE\*{0,2}(?:\s+(?:\d+\s+){3}|<br>)(\d+)\b"
)


def normalize_flow_tables(block: str) -> str:
    """Wandelt als Fließtext extrahierte Einheiten-Tabellen in Pipe-Tabellenzeilen um."""

    def repl(m: re.Match) -> str:
        rows = [f"|{m.group(1)}|{m.group(2)}|{m.group(3)}|"]
        if m.group(4):
            rows.append(f"|{m.group(4)}|||")
        return "\n".join(rows)

    block = SPLIT_UNITS_RE.sub(lambda m: f"|{m.group(1)}|{m.group(2)}|{m.group(3)}|", block)
    return FLOW_UNITS_RE.sub(repl, block)


def split_modules(text: str) -> list[tuple[str, str, str]]:
    """Liefert (Name, Nummer, Block) je W3M-Modul; ein Block endet am nächsten `## `-Titel.

    Steht der englische Titel in der `## `-Zeile (statt in einer `### `-Zeile), wird er dem Block
    als `### `-Zeile vorangestellt.
    """
    next_h2 = re.compile(r"^## ", re.M)
    matches = list(MODULE_RE.finditer(text))
    result = []
    for m in matches:
        nxt = next_h2.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        block = normalize_flow_tables(text[m.end():end])
        if m.group(3):
            block = f"\n### **{m.group(3).strip()}**\n{block}"
        result.append((m.group(1).strip(), m.group(2), block))
    return result


def sections(block: str) -> list[tuple[str, str]]:
    """Zerlegt einen Modulblock in (Überschrift, Text)-Paare entlang der `#`-Überschriften."""
    heads = list(HEADING_RE.finditer(block))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(block)
        out.append((h.group(1).strip(), block[h.end():end]))
    return out


def text_before_table(raw: str) -> str:
    """Abschnittstext bis zur ersten Tabellenzeile (Tabelle `LERNEINHEITEN UND INHALTE`)."""
    kept = []
    for line in raw.splitlines():
        if line.startswith("|"):
            break
        kept.append(line)
    return "\n".join(kept)


def parse_table(raw: str) -> tuple[list[str], list[str]]:
    """Zerlegt die Tabelle `LERNEINHEITEN UND INHALTE` in (Einheiten-Titel, Inhaltszeilen).

    Zeilen mit Zahlen in Spalte 2 und 3 (Präsenzzeit/Selbststudium) sind Titel der Lehr- und
    Lerneinheiten; Zeilen ohne Stundenangaben enthalten die Inhalte.
    """
    units, content = [], []
    for line in raw.splitlines():
        if not line.startswith("|") or set(line) <= set("|-"):
            continue
        cells = line.strip().strip("|").split("|")
        first = cells[0]
        if "LERNEINHEITEN UND INHALTE" in first.upper():
            continue
        has_hours = len(cells) >= 3 and all(re.search(r"\d", c) for c in cells[1:3])
        if "LEHR- UND LERNEINHEITEN" in first.upper():
            if "<br>" not in first:
                continue  # reine Kopfzeile
            first = first.split("<br>", 1)[1]  # Layout 1: Überschrift und Titel in einer Zelle
            has_hours = True
        first = clean(first)
        if not first:
            continue
        (units if has_hours else content).append(first)
    return units, content


def text_after_table(raw: str) -> str:
    """Fließtext hinter der Tabelle (Inhalte stehen teils dort statt in einer Tabellenzeile)."""
    lines = raw.splitlines()
    last = max((i for i, ln in enumerate(lines) if ln.startswith("|")), default=-1)
    return "\n".join(lines[last + 1:]) if last >= 0 else ""


def metadata(block: str) -> tuple[str, str]:
    """(ECTS, Modulverantwortung) aus den formalen Angaben."""
    ects = ""
    flow = FLOW_ECTS_RE.search(block)
    if flow:
        ects = flow.group(1)
    rows = [] if ects else block.splitlines()
    for i, line in enumerate(rows):
        if "ECTS-LEISTUNGSPUNKTE" in line and "DAVON SELBSTSTUDIUM" in line:
            for nxt in rows[i + 1:i + 4]:
                cells = [c for c in nxt.strip().strip("|").split("|") if c.strip()]
                if cells and re.fullmatch(r"\d+([.,]\d+)?", cells[-1].strip()):
                    ects = cells[-1].strip()
                    break
            break
    head = block.split("QUALIFIKATIONSZIELE")[0]
    cell = re.search(r"\|\s*(Prof[^|<]+?)\s*\|", head)
    if cell:
        return ects, clean(cell.group(1))
    head = re.sub(r"<br>|\|", " ", head).replace("**", "")
    m = re.search(
        r"((?:Prof\.?|Dr\.?)\s*(?:\.?\s*)?(?:Dr\.?\s*)?(?:[\w.\-]+\s+)*?[\wÄÖÜäöüß\-]+)"
        r"(?=\s+(?:Deutsch|Englisch|Französisch|Spanisch)|\s*$|\s+EINGESETZTE|\s+SPRACHE|\s+\d)",
        head,
    )
    return ects, clean(m.group(1)) if m else ""


def parse_module(name: str, number: str, block: str) -> dict[str, str]:
    secs = sections(block)
    row = {f: "" for f in FIELDS}
    row["modulnummer"] = number
    row["modulname"] = name

    en = re.search(r"^### \*\*(.+?)\*\*\s*$", block, re.M)
    row["modulname_en"] = clean(en.group(1)) if en else ""
    row["ects"], row["verantwortung"] = metadata(block)

    last_comp_idx = -1
    for i, (title, raw) in enumerate(secs):
        key = COMPETENCES.get(title.upper())
        if key:
            row[key] = clean(text_before_table(raw))
            last_comp_idx = i

    # Lehreinheiten-Titel und Inhalte: Tabelle + Fließtext direkt nach den Kompetenzen
    units: list[str] = []
    content: list[str] = []
    if last_comp_idx >= 0:
        raw = secs[last_comp_idx][1]
        u, c = parse_table(raw)
        units += u
        content += c
        content.append(text_after_table(raw))
    for title, raw in secs[last_comp_idx + 1:]:
        if title.upper() in {"BESONDERHEITEN", "VORAUSSETZUNGEN", "LITERATUR"}:
            break
        u, c = parse_table(raw)
        units += u
        content += c
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("|"))
        if title.upper() not in {"LERNEINHEITEN UND INHALTE", "LEHR- UND LERNEINHEITEN"}:
            raw = f"{title}\n{raw}"  # Zwischenüberschriften der Inhalte erhalten
        content.append(raw)
    row["lehr_lerneinheiten"] = "\n".join(dict.fromkeys(units))
    row["inhalte"] = clean("\n".join(content))

    for title, raw in secs:
        if title.upper() == "VORAUSSETZUNGEN":
            row["voraussetzungen"] = clean(raw)
        elif title.upper() == "LITERATUR":
            row["literatur"] = clean(raw)

    row["text_gesamt"] = "\n".join(
        part
        for part in (
            row["fachkompetenz"],
            row["methodenkompetenz"],
            row["personale_soziale_kompetenz"],
            row["uebergreifende_handlungskompetenz"],
            row["lehr_lerneinheiten"],
            row["inhalte"],
        )
        if part
    )
    return row


def extract(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> int:
    text = input_path.read_text(encoding="utf-8")
    rows = [parse_module(*m) for m in split_modules(text)]
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    n = extract()
    print(f"{n} Module nach {DEFAULT_OUTPUT} geschrieben.")


if __name__ == "__main__":
    main()
