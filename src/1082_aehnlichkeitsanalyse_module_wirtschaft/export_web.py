"""Exportiert die Ähnlichkeitsergebnisse als `web/data/data.json` für die Website.

Vorab berechnet (die Website rechnet nichts Schweres selbst):
- Zusammenlegungsfolge je Größengrenze: agglomeratives Clustering mit Average Linkage auf dem
  Gesamt-Ähnlichkeitswert (z), wobei nur Cluster bis `cap` Module verschmelzen dürfen.
  Die Website spielt die ersten `n − k` Schritte ab, um `k` Cluster zu erhalten.
- Blattreihenfolge je Größengrenze für die Matrixansicht (jedes Cluster ist darin zusammenhängend).
- 2D-Karte per UMAP auf dem Abstand `1 − Perzentil(Gesamtähnlichkeit)`.
- Je Modul die gewichteten Fachbegriffe (TF-IDF) zur Benennung von Clustern.
- Begründungen (ähnlichste Stichpunkte, geteilte Literatur) für alle hinreichend ähnlichen Paare.

Modulverantwortliche werden bewusst nicht exportiert.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .similarity import STOPWORDS_DE

ROOT = Path(__file__).resolve().parents[2]
PREP_DIR = ROOT / "data" / "aufbereitet"
SIM_DIR = ROOT / "data" / "aehnlichkeit"
OUT_PATH = ROOT / "web" / "data" / "data.json"

# Rahmenmodule: ähnlich formuliert, aber fachlich nicht verwandt
RAHMENMODULE = {
    "W3M10011", "W3M10012", "W3M10013",  # Forschungsprojektarbeit I/II, Masterarbeit
    "W3M12001", "W3M12002",  # Projektarbeit I/II
    "W3M20031", "W3M20032", "W3M20033", "W3M20040",  # Forschungsprojektarbeit WI I/II, Studienarbeit, Masterarbeit
    "W3M50009", "W3M50010",  # Studienarbeit, Masterarbeit
}

# Studiengang nach Präfix der Modulnummer (längstes passendes Präfix gewinnt); Namen wie auf cas.dhbw.de
STUDIENGAENGE = {
    "W3M101": ("ACT", "Accounting, Controlling, Taxation"),
    "W3M113": ("DBM", "Digital Business Management"),
    "W3M5": ("DSAI", "Data Science and Artificial Intelligence"),
    "W3M114": ("ENT", "Entrepreneurship"),
    "W3M104": ("FIN", "Finance"),
    "W3M102": ("GBM", "General Business Management"),
    "W3M107": ("MKT", "Marketing and Business Psychology"),
    "W3M4": ("MBA", "Master of Business Administration"),
    "W3M108": ("MDB", "Media and Data-driven Business"),
    "W3M109": ("PMW", "Personalmanagement und Wirtschaftspsychologie"),
    "W3M112": ("SAL", "Sales and Negotiation"),
    "W3M110": ("SLP", "Supply Chain Management, Logistics, Production"),
    "W3M2": ("WI", "Wirtschaftsinformatik"),
}
STUDIENGANG_SONST = ("ÜG", "Studiengangsübergreifend")

CAPS = [3, 4, 5, 6, 8, 0]  # 0 = ohne Grenze
EVIDENCE_MIN_Z = 1.0
TOP_TERMS = 25
MAX_POINTS = 14

# Begriffe, die in fast jedem Modul vorkommen und keine Fachfamilie kennzeichnen
GENERIC_TERMS = set("""
studierenden grundlagen einführung überblick aktuelle ausgewählte rahmen bedeutung methoden instrumente
modul themen aspekte anwendung anwendungen praxis beispiele fallstudien konzepte ansätze möglichkeiten
überblick vertiefung herausforderungen grenzen ziele aufgaben unternehmen management
studierende fähigkeiten erfolgt erwerben dabei gezielt erarbeitet erarbeiten vergleich externen zwecke
werden wurde sowie insbesondere jeweiligen verschiedene verschiedenen wesentliche wesentlichen relevante
relevanten moderne modernen neue neuen eigene eigenen etwa hierbei hinblick stehen steht liegt
bereich bereiche bereichen verfahren systematisierung persönliche theorie theorien konzept modell modelle
modellen rolle form formen fragen fragestellungen basis entwicklung entwicklungen umsetzung analyse analysen
gestaltung einsatz arten überblick teil teile kontext perspektive perspektiven erfolg faktoren anforderungen
""".split())


ADJECTIVE_ENDINGS = ("ische", "ischen", "isches", "ale", "alen", "ales", "liche", "lichen", "liches", "ige", "igen",
                     "ive", "iven", "ierte", "ierten", "ende", "enden", "ete", "eten", "orientierte", "bezogene")


def load_matrix(name: str, ids: list[str]) -> np.ndarray:
    m = pd.read_csv(SIM_DIR / f"matrix_{name}.csv", index_col=0)
    m.index = m.index.astype(str)
    return m.loc[ids, ids].to_numpy(dtype=float)


def studiengang(modulnummer: str) -> tuple[str, str]:
    prefixes = [p for p in STUDIENGAENGE if modulnummer.startswith(p)]
    return STUDIENGAENGE[max(prefixes, key=len)] if prefixes else STUDIENGANG_SONST


def capped_average_linkage(sim: np.ndarray, cap: int) -> list[list]:
    """Greedy Average Linkage; Rückgabe: Liste [a, b, ähnlichkeit] mit Cluster-IDs im SciPy-Schema."""
    n = len(sim)
    s = np.where(np.isnan(sim), -np.inf, sim).astype(float)
    sizes = {i: 1 for i in range(n)}
    active = list(range(n))
    # Summen der paarweisen Ähnlichkeiten zwischen aktiven Clustern
    total = {(i, j): s[i, j] for i in range(n) for j in range(i + 1, n)}
    members = {i: [i] for i in range(n)}
    merges = []
    next_id = n
    while len(active) > 1:
        best, best_pair = -np.inf, None
        for x_idx, a in enumerate(active):
            for b in active[x_idx + 1:]:
                if cap and sizes[a] + sizes[b] > cap:
                    continue
                key = (a, b) if a < b else (b, a)
                avg = total[key] / (sizes[a] * sizes[b])
                if avg > best:
                    best, best_pair = avg, (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        c = next_id
        next_id += 1
        sizes[c] = sizes[a] + sizes[b]
        members[c] = members[a] + members[b]
        active = [x for x in active if x not in (a, b)]
        for x in active:
            ka = (min(a, x), max(a, x))
            kb = (min(b, x), max(b, x))
            total[(x, c)] = total.pop(ka) + total.pop(kb)
        total.pop((min(a, b), max(a, b)), None)
        active.append(c)
        merges.append([a, b, round(float(best), 3)])
    return merges


def leaf_order(n: int, merges: list[list], base_order: list[int]) -> list[int]:
    """Blattreihenfolge des Merge-Walds; Wurzeln nach ihrer Position in `base_order` sortiert."""
    children = {n + k: (a, b) for k, (a, b, _) in enumerate(merges)}
    merged = {a for a, _, _ in merges} | {b for _, b, _ in merges}
    roots = [c for c in list(range(n)) + list(children) if c not in merged]
    rank = {leaf: r for r, leaf in enumerate(base_order)}

    def leaves(c: int) -> list[int]:
        if c < n:
            return [c]
        a, b = children[c]
        la, lb = leaves(a), leaves(b)
        return la + lb if min(rank[x] for x in la) <= min(rank[x] for x in lb) else lb + la

    out = []
    for root in sorted(roots, key=lambda r: min(rank[x] for x in leaves(r))):
        out.extend(leaves(root))
    return out


def umap_coords(total: np.ndarray) -> np.ndarray:
    import umap

    iu = np.triu_indices_from(total, k=1)
    pct = pd.Series(total[iu]).rank(pct=True).to_numpy()
    d = np.zeros_like(total)
    d[iu] = 1 - pct
    d = d + d.T
    reducer = umap.UMAP(n_neighbors=8, min_dist=0.35, spread=1.2, metric="precomputed", random_state=42)
    xy = reducer.fit_transform(d)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    return xy


def module_terms(modules: pd.DataFrame) -> list[list]:
    texts = (modules["modulname"] + ". " + modules["modulname"] + ". " + modules["text_inhalte"]).tolist()
    vec = TfidfVectorizer(lowercase=True, stop_words=STOPWORDS_DE + sorted(GENERIC_TERMS), ngram_range=(1, 2),
                          min_df=2, max_df=0.3, sublinear_tf=True, token_pattern=r"(?u)\b[^\W\d_][\w&-]{2,}\b")
    x = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    # Häufigste Originalschreibweise je Wort (für die Anzeige: „IFRS“, „Supply Chain“)
    forms: dict[str, Counter] = {}
    for word in re.findall(r"(?u)\b[^\W\d_][\w&-]{2,}\b", " ".join(texts)):
        forms.setdefault(word.lower(), Counter())[word] += 1

    def display(term: str) -> str:
        return " ".join(forms[w].most_common(1)[0][0] if w in forms else w for w in term.split())

    # Dokumenthäufigkeit je Term, um feste Wortpaare („Supply Chain“) von Zufallspaaren zu trennen
    doc_freq = dict(zip(vocab, np.asarray((x > 0).sum(axis=0)).ravel()))

    def looks_adjective(word: str) -> bool:
        return word.lower().endswith(ADJECTIVE_ENDINGS)

    def is_label_term(term: str) -> bool:
        # Nur Substantive bzw. Abkürzungen (im Deutschen großgeschrieben), keine Füllwörter wie „diesem“
        words = display(term).split()
        if not all(w[0].isupper() and w.lower() not in GENERIC_TERMS for w in words):
            return False
        if len(words) == 1:
            return not looks_adjective(words[0])
        a, b = term.split()
        together = doc_freq[term] >= 3 and doc_freq[term] >= 0.6 * min(doc_freq.get(a, 1e9), doc_freq.get(b, 1e9))
        return together and not looks_adjective(words[1])

    out = []
    for i in range(x.shape[0]):
        row = x.getrow(i)
        idx = [k for k in row.indices[np.argsort(row.data)[::-1]] if is_label_term(vocab[k])][:TOP_TERMS]
        out.append([[vocab[k], round(float(row[0, k]), 3), display(vocab[k])] for k in idx])
    return out


def parse_matches(text: str) -> list[dict]:
    if not isinstance(text, str) or not text:
        return []
    out = []
    for part in text.split(" || "):
        m = re.match(r"\[(\d\.\d+)\] (.*) ⟷ (.*)", part, re.S)
        if m:
            out.append({"s": float(m.group(1)), "a": m.group(2), "b": m.group(3)})
    return out


def export() -> dict:
    modules = pd.read_csv(PREP_DIR / "module.csv", dtype=str, keep_default_na=False)
    modules = modules[~modules["modulnummer"].isin(RAHMENMODULE)].reset_index(drop=True)
    points = pd.read_csv(PREP_DIR / "inhalte_stichpunkte.csv", dtype={"modulnummer": str})
    lit = pd.read_csv(PREP_DIR / "literatur.csv", dtype={"modulnummer": str})
    ids = modules["modulnummer"].tolist()
    pos = {mid: i for i, mid in enumerate(ids)}
    n = len(ids)

    total = load_matrix("gesamt", ids)
    dims = {k: load_matrix(k, ids) for k in ("inhalte", "kompetenzen", "literatur")}

    # Basisreihenfolge: ungebremstes Average Linkage (für stabile Matrixsortierung)
    unconstrained = capped_average_linkage(total, 0)
    base = leaf_order(n, unconstrained, list(range(n)))
    merges = {str(cap): (unconstrained if cap == 0 else capped_average_linkage(total, cap)) for cap in CAPS}
    orders = {cap: leaf_order(n, m, base) for cap, m in merges.items()}

    xy = umap_coords(total)
    terms = module_terms(modules)

    mod_out = []
    for i, r in modules.iterrows():
        pts = points[points["modulnummer"] == r["modulnummer"]]
        heads = [h for h in pts["zwischenueberschrift"].dropna().unique().tolist() if str(h).strip()]
        items = pts["stichpunkt"].tolist()
        lit_rows = lit[(lit["modulnummer"] == r["modulnummer"]) & lit["lit_key"].notna()]
        mod_out.append({
            "id": r["modulnummer"],
            "name": r["modulname"],
            "name_en": r["modulname_en"],
            "sg": studiengang(r["modulnummer"])[0],
            "ects": int(float(r["ects"])) if r["ects"] else None,
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "terms": terms[i],
            "headings": heads[:8],
            "points": items[:MAX_POINTS],
            "points_total": len(items),
            "lit": [[k, f"{a}: {t}"] for k, a, t in
                    zip(lit_rows["lit_key"], lit_rows["erstautor_nachname"].fillna(""), lit_rows["titel"].fillna(""))],
        })

    pairs = pd.read_csv(SIM_DIR / "paare.csv", dtype={"modul_a": str, "modul_b": str})
    pairs = pairs[pairs["modul_a"].isin(pos) & pairs["modul_b"].isin(pos)]
    evidence = {}
    for _, r in pairs[pairs["gesamt_z"] >= EVIDENCE_MIN_Z].iterrows():
        a, b = pos[r["modul_a"]], pos[r["modul_b"]]
        key = f"{min(a, b)}-{max(a, b)}"
        swap = a > b
        matches = parse_matches(r["aehnlichste_stichpunkte"])
        if swap:
            matches = [{"s": m["s"], "a": m["b"], "b": m["a"]} for m in matches]
        evidence[key] = {"m": matches, "l": r["gemeinsame_literatur"] if isinstance(r["gemeinsame_literatur"], str) else ""}

    def rounded(m: np.ndarray) -> list[list]:
        return [[None if np.isnan(v) else round(float(v), 2) for v in row] for row in m]

    iu = np.triu_indices(n, k=1)
    data = {
        "meta": {
            "n": n,
            "n_total": n + len(RAHMENMODULE),
            "excluded": sorted(RAHMENMODULE),
            "caps": CAPS,
            "studiengaenge": dict(sorted(set(STUDIENGAENGE.values()) | {STUDIENGANG_SONST})),
            "weights": {"inhalte": 0.45, "kompetenzen": 0.35, "literatur": 0.20},
            # Verteilung aller Paarwerte → Einordnung „gehört zu den ähnlichsten x %“
            "quantiles": [round(float(q), 3) for q in np.quantile(total[iu], np.linspace(0, 1, 1001))],
        },
        "modules": mod_out,
        "sim": rounded(total),
        "dims": {k: rounded(v) for k, v in dims.items()},
        "merges": merges,
        "orders": {str(k): v for k, v in orders.items()},
        "evidence": evidence,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"n": n, "merges": {k: len(v) for k, v in merges.items()}, "evidence": len(evidence),
            "bytes": OUT_PATH.stat().st_size}


def main() -> None:
    stats = export()
    print(f"{stats['n']} Module exportiert → {OUT_PATH} ({stats['bytes'] / 1e6:.1f} MB)")
    print("Erreichbare Mindestanzahl Cluster je Größengrenze:",
          {k: stats["n"] - v for k, v in stats["merges"].items()})
    print(f"Begründungen für {stats['evidence']} Paare")


if __name__ == "__main__":
    main()
