"""Exportiert die Modulauslastung als `web/data/auslastung.json` für die Website (Punktgröße, Cluster-Summen).

Eingabe ist die neueste `data/*Modulauslastung*.xlsx` (Blatt „Datenbasis“): je Durchführung eine Zeile mit
Modulcode, maximaler Teilnehmerzahl und Teilnehmenden je Semester. Ein Modul kann mehrere Zeilen haben
(parallele Gruppen, z. B. Forschungsmethoden je Studiengang, oder Vorgänger unter gleichem Code) – sie werden
addiert. „X“ in einer Semesterspalte steht für eine abgesagte Durchführung und zählt als 0 Teilnehmende.

Kennzahl für die Karte: Ø Teilnehmende pro Studienjahr (WiSe + folgendes SoSe) über `STUDIENJAHRE`.
Ein Mittel über mehrere Jahre glättet Wahlmodule, die nicht jedes Jahr laufen; das laufende Semester ist
noch unvollständig und bleibt außen vor.

Die Datei liegt bewusst getrennt von `data.json`: Die Website funktioniert auch ohne sie (dann ohne Auslastung).
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODULES_PATH = ROOT / "web" / "data" / "data.json"
OUT_PATH = ROOT / "web" / "data" / "auslastung.json"

STUDIENJAHRE = ["23/24", "24/25", "25/26"]
CANCELLED = "X"


def latest_source() -> Path:
    files = sorted(DATA_DIR.glob("*Modulauslastung*.xlsx"))
    if not files:
        raise FileNotFoundError(f"Keine *Modulauslastung*.xlsx in {DATA_DIR}")
    # Datum im Dateinamen (…_JJJJMMTT.xlsx) entscheidet, sonst alphabetisch
    return max(files, key=lambda p: (re.findall(r"(\d{8})", p.stem) or [""])[-1] + p.name)


def semesters(jahr: str) -> tuple[str, str]:
    """„23/24“ → („WiSe23/24“, „SoSe24“)."""
    return f"WiSe{jahr}", f"SoSe{jahr.split('/')[1]}"


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Datenbasis", dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    needed = ["Modulcode", "max TN", *(s for j in STUDIENJAHRE for s in semesters(j))]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen in {path.name}: {missing}")
    df["Modulcode"] = df["Modulcode"].astype(str).str.strip()
    return df


def module_stats(rows: pd.DataFrame) -> dict:
    per_year, held, cancelled, participants = [], 0, 0, 0
    for jahr in STUDIENJAHRE:
        total = 0
        for col in semesters(jahr):
            for v in rows[col]:
                if isinstance(v, str) and v.strip().upper() == CANCELLED:
                    cancelled += 1
                elif pd.notna(v) and str(v).strip():
                    total += int(float(v))
                    held += 1
        per_year.append(total)
        participants += total
    # Höchstteilnehmerzahl nur aus Durchführungen im Zeitraum (Vorgänger unter gleichem Code haben teils andere)
    window = [s for j in STUDIENJAHRE for s in semesters(j)]
    active = rows[rows[window].notna().any(axis=1)]
    max_tn = pd.to_numeric(active["max TN"], errors="coerce").max()
    per_run = participants / held if held else None
    return {
        "jahr": round(participants / len(STUDIENJAHRE), 1),
        "jahre": per_year,
        "durchf": held,
        "abgesagt": cancelled,
        "je_durchf": None if per_run is None else round(per_run, 1),
        "max": None if pd.isna(max_tn) else int(max_tn),
        "quote": None if per_run is None or pd.isna(max_tn) else round(per_run / max_tn, 3),
    }


def export() -> dict:
    src = latest_source()
    df = read_source(src)
    ids = [m["id"] for m in json.loads(MODULES_PATH.read_text(encoding="utf-8"))["modules"]]
    groups = {code: g for code, g in df.groupby("Modulcode") if code in set(ids)}
    stand = re.findall(r"(\d{4})(\d{2})(\d{2})", src.stem)
    data = {
        "meta": {
            "quelle": src.name,
            "stand": f"{stand[-1][2]}.{stand[-1][1]}.{stand[-1][0]}" if stand else None,
            "studienjahre": STUDIENJAHRE,
        },
        "module": {mid: module_stats(groups[mid]) for mid in ids if mid in groups},
    }
    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"src": src.name, "found": len(data["module"]), "n": len(ids),
            "missing": [mid for mid in ids if mid not in groups]}


def main() -> None:
    stats = export()
    print(f"Auslastung aus {stats['src']}: {stats['found']} von {stats['n']} Modulen → {OUT_PATH}")
    if stats["missing"]:
        print("Ohne Auslastungsdaten:", ", ".join(stats["missing"]))


if __name__ == "__main__":
    main()
