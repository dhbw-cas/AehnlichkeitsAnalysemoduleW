"""Stufe 1 der Ähnlichkeitsanalyse: paarweise Modulähnlichkeit in drei Dimensionen.

Eingabe sind die Tabellen aus `data/aufbereitet/` (siehe `preprocess.py`).

Dimensionen:
1. Inhalte     – Mischung aus
                 a) Embedding des gesamten Inhaltstextes (Kosinus),
                 b) Abgleich auf Stichpunkt-Ebene (symmetrisches Mittel der besten Treffer, à la BERTScore),
                 c) TF-IDF-Kosinus (exakte Fachbegriffe wie HGB, IFRS, SPSS).
2. Kompetenzen – Embedding der um Standardformulierungen bereinigten Kompetenztexte.
3. Literatur   – IDF-gewichteter Dice-Koeffizient über unscharf zusammengeführte `lit_key`s
                 (seltene geteilte Werke zählen mehr als Standardwerke wie „Wissenschaftliches Arbeiten“).

Inhalte und Kompetenzen werden über alle Modulpaare z-standardisiert, damit Skalen vergleichbar werden.
Die Literatur nicht: nur ~1 % der Paare teilt überhaupt ein Werk, z-Werte würden bis ~40 explodieren und
den Gesamtwert dominieren. Sie geht daher als Bonus `LIT_SCALE × Dice` ein (ein geteiltes Buch ≈ +0,5,
vier geteilte Werke ≈ +2,6 – vergleichbar mit dem obersten Prozent bei Inhalten/Kompetenzen).
Der Gesamtwert ist das gewichtete Mittel der Dimensions-z-Werte; fehlt einem Modul die Literatur,
werden die übrigen Gewichte renormiert.

Ausgaben in `data/aehnlichkeit/`:
- `embeddings_{inhalte,kompetenzen}.npy` (+ `embeddings_index.csv`) für UMAP/HDBSCAN,
- `embeddings_stichpunkte.npy` (Zeilenreihenfolge wie `inhalte_stichpunkte.csv`),
- `matrix_{inhalte,kompetenzen,literatur,gesamt}.csv` (169×169, z-Werte bzw. Literatur-Bonus; Diagonale leer),
- `paare.csv`: alle Paare mit Rohwerten, z-Werten, Perzentil, geteilter Literatur und
  den am besten übereinstimmenden Stichpunkten (Begründung),
- `top_n.csv`: je Modul die ähnlichsten Module.
"""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "aufbereitet"
DEFAULT_OUTPUT = ROOT / "data" / "aehnlichkeit"

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 2048

WEIGHTS = {"inhalte": 0.45, "kompetenzen": 0.35, "literatur": 0.20}
CONTENT_MIX = {"dokument": 0.4, "stichpunkte": 0.4, "tfidf": 0.2}
LIT_FUZZY_THRESHOLD = 88
LIT_SCALE = 6.0
TOP_N = 10
TOP_POINT_MATCHES = 3

# Häufige Funktionswörter für TF-IDF (Embeddings brauchen keine Stoppwortliste)
STOPWORDS_DE = """aber alle als also am an auch auf aus bei bis bzw da damit dann das dass dem den der des die
dies diese dieser dieses durch ein eine einem einen einer eines es etc für ggf hat im in ist je kann können mit
nach nicht noch nur oder sich sie so sowie sowohl über um und unter usw vom von vor wie wird werden z zu zum zur
zwischen b the of and to in for on with a an""".split()


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def zscore_offdiag(m: np.ndarray) -> np.ndarray:
    """z-Standardisierung über alle Paare (obere Dreiecksmatrix, NaN ignoriert)."""
    iu = np.triu_indices_from(m, k=1)
    vals = m[iu]
    mu, sd = np.nanmean(vals), np.nanstd(vals)
    z = (m - mu) / (sd if sd > 0 else 1.0)
    np.fill_diagonal(z, np.nan)
    return z


def cosine_matrix(emb: np.ndarray) -> np.ndarray:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb @ emb.T


def text_hash(texts: list[str]) -> str:
    return hashlib.sha256("\x1f".join([MODEL_NAME, *texts]).encode()).hexdigest()[:16]


class Embedder:
    """Lädt das Modell erst bei Bedarf; Ergebnisse werden je Textliste per Hash zwischengespeichert."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._model = None

    def encode(self, name: str, texts: list[str]) -> np.ndarray:
        path = self.cache_dir / f"embeddings_{name}.npy"
        stamp = self.cache_dir / f"embeddings_{name}.hash"
        h = text_hash(texts)
        if path.exists() and stamp.exists() and stamp.read_text().strip() == h:
            return np.load(path)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME, device="cpu")
            self._model.max_seq_length = MAX_SEQ_LENGTH
        # Lange Texte zuerst gruppieren beschleunigt das Batching; Reihenfolge wird wiederhergestellt
        order = np.argsort([-len(t) for t in texts])
        enc = self._model.encode([texts[i] for i in order], batch_size=8, normalize_embeddings=True,
                                 show_progress_bar=True)
        emb = np.empty_like(enc)
        emb[order] = enc
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        stamp.write_text(h)
        return emb


# ---------------------------------------------------------------------------
# 1. Inhalte
# ---------------------------------------------------------------------------


def point_text(row: pd.Series) -> str:
    head = row["zwischenueberschrift"]
    return f"{head}: {row['stichpunkt']}" if isinstance(head, str) and head.strip() else row["stichpunkt"]


def point_alignment(points: pd.DataFrame, emb_points: np.ndarray, ids: list[str]):
    """Symmetrisches Mittel der besten Stichpunkt-Treffer je Modulpaar.

    Liefert die Ähnlichkeitsmatrix und je Paar die stärksten Stichpunkt-Übereinstimmungen.
    """
    sim = emb_points @ emb_points.T
    groups = {mid: np.flatnonzero(points["modulnummer"].to_numpy() == mid) for mid in ids}
    n = len(ids)
    out = np.full((n, n), np.nan)
    best: dict[tuple[int, int], list[tuple[float, int, int]]] = {}
    for i in range(n):
        gi = groups[ids[i]]
        for j in range(i + 1, n):
            gj = groups[ids[j]]
            if len(gi) == 0 or len(gj) == 0:
                continue
            block = sim[np.ix_(gi, gj)]
            score = 0.5 * (block.max(axis=1).mean() + block.max(axis=0).mean())
            out[i, j] = out[j, i] = score
            # Stärkste Einzeltreffer (jeder Stichpunkt höchstens einmal) als Begründung
            flat_idx = np.argsort(block, axis=None)[::-1]
            used_a, used_b, matches = set(), set(), []
            for k in flat_idx:
                a, b = divmod(int(k), block.shape[1])
                if a in used_a or b in used_b:
                    continue
                matches.append((float(block[a, b]), int(gi[a]), int(gj[b])))
                used_a.add(a)
                used_b.add(b)
                if len(matches) == TOP_POINT_MATCHES:
                    break
            best[(i, j)] = matches
    return out, best


def tfidf_matrix(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(lowercase=True, stop_words=STOPWORDS_DE, ngram_range=(1, 2), min_df=2,
                          max_df=0.5, sublinear_tf=True, token_pattern=r"(?u)\b[\w&-]{2,}\b")
    x = vec.fit_transform(texts)
    return (x @ x.T).toarray()


# ---------------------------------------------------------------------------
# 3. Literatur
# ---------------------------------------------------------------------------


def canonical_lit_keys(keys: list[str]) -> dict[str, str]:
    """Führt Schreibvarianten zusammen: gleicher Erstautor und sehr ähnlicher Titelteil."""
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_author: dict[str, list[str]] = {}
    for k in keys:
        by_author.setdefault(k.split("_", 1)[0], []).append(k)
    for group in by_author.values():
        group.sort()
        for a_idx, a in enumerate(group):
            ta = a.split("_", 1)[1] if "_" in a else ""
            for b in group[a_idx + 1:]:
                tb = b.split("_", 1)[1] if "_" in b else ""
                if ta and tb and fuzz.ratio(ta, tb) >= LIT_FUZZY_THRESHOLD:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    return {k: find(k) for k in keys}


def literature_matrix(lit: pd.DataFrame, ids: list[str]):
    lit = lit.dropna(subset=["lit_key"])
    canon = canonical_lit_keys(sorted(lit["lit_key"].unique()))
    sets = {mid: set() for mid in ids}
    for mid, key in zip(lit["modulnummer"], lit["lit_key"]):
        if mid in sets:
            sets[mid].add(canon[key])
    n_with = sum(1 for s in sets.values() if s)
    df: dict[str, int] = {}
    for s in sets.values():
        for k in s:
            df[k] = df.get(k, 0) + 1
    idf = {k: math.log(n_with / d) + 1.0 for k, d in df.items()}

    n = len(ids)
    out = np.full((n, n), np.nan)
    shared: dict[tuple[int, int], list[str]] = {}
    for i in range(n):
        si = sets[ids[i]]
        wi = sum(idf[k] for k in si)
        for j in range(i + 1, n):
            sj = sets[ids[j]]
            if not si or not sj:
                continue
            common = si & sj
            wj = sum(idf[k] for k in sj)
            out[i, j] = out[j, i] = 2 * sum(idf[k] for k in common) / (wi + wj)
            if common:
                shared[(i, j)] = sorted(common, key=lambda k: (df[k], k))
    return out, shared, canon


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    df.to_csv(path, index=index, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\r\n")


def combine(zs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    num = np.zeros_like(next(iter(zs.values())))
    den = np.zeros_like(num)
    for name, z in zs.items():
        ok = ~np.isnan(z)
        num[ok] += weights[name] * z[ok]
        den[ok] += weights[name]
    with np.errstate(invalid="ignore"):
        out = num / den
    np.fill_diagonal(out, np.nan)
    return out


def percentile_matrix(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(m, k=1)
    ranks = pd.Series(m[iu]).rank(pct=True).to_numpy()
    out = np.full_like(m, np.nan)
    out[iu] = ranks
    out.T[iu] = ranks
    return out


def compute(input_dir: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    modules = pd.read_csv(input_dir / "module.csv", dtype=str, keep_default_na=False)
    points = pd.read_csv(input_dir / "inhalte_stichpunkte.csv", dtype={"modulnummer": str})
    lit = pd.read_csv(input_dir / "literatur.csv", dtype={"modulnummer": str})
    ids = modules["modulnummer"].tolist()
    names = dict(zip(modules["modulnummer"], modules["modulname"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(output_dir)

    # Embeddings
    emb_content = embedder.encode("inhalte", modules["text_inhalte"].tolist())
    emb_comp = embedder.encode("kompetenzen", modules["text_kompetenzen"].tolist())
    point_texts = [point_text(r) for _, r in points.iterrows()]
    emb_points = embedder.encode("stichpunkte", point_texts)
    write_csv(modules[["modulnummer", "modulname"]], output_dir / "embeddings_index.csv")

    # Rohwerte je Komponente
    raw = {
        "inhalte_dokument": cosine_matrix(emb_content),
        "inhalte_tfidf": tfidf_matrix(modules["text_inhalte"].tolist()),
        "kompetenzen": cosine_matrix(emb_comp),
    }
    raw["inhalte_stichpunkte"], point_matches = point_alignment(points, emb_points, ids)
    raw["literatur"], shared_lit, canon = literature_matrix(lit, ids)
    for m in raw.values():
        np.fill_diagonal(m, np.nan)

    # Standardisieren und zu Dimensionen zusammenfassen
    z = {k: zscore_offdiag(v) for k, v in raw.items()}
    z_content = combine({f"inhalte_{k}": z[f"inhalte_{k}"] for k in CONTENT_MIX},
                        {f"inhalte_{k}": w for k, w in CONTENT_MIX.items()})
    dims = {
        "inhalte": zscore_offdiag(z_content),
        "kompetenzen": z["kompetenzen"],
        "literatur": LIT_SCALE * raw["literatur"],
    }
    total = combine(dims, WEIGHTS)
    pct = percentile_matrix(total)

    for name, m in {**dims, "gesamt": total}.items():
        write_csv(pd.DataFrame(np.round(m, 4), index=ids, columns=ids).rename_axis("modulnummer"),
                  output_dir / f"matrix_{name}.csv", index=True)

    # Lange Paartabelle mit Begründung
    key_title = lit.dropna(subset=["lit_key"]).drop_duplicates("lit_key").set_index("lit_key")
    def lit_label(k: str) -> str:
        if k in key_title.index:
            r = key_title.loc[k]
            return f"{r['erstautor_nachname']}: {r['titel']}"
        return k

    rows = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            matches = point_matches.get((i, j), [])
            rows.append({
                "modul_a": ids[i], "name_a": names[ids[i]],
                "modul_b": ids[j], "name_b": names[ids[j]],
                "gesamt_z": total[i, j], "gesamt_perzentil": pct[i, j],
                "inhalte_z": dims["inhalte"][i, j], "kompetenzen_z": dims["kompetenzen"][i, j],
                "literatur_bonus": dims["literatur"][i, j],
                "inhalte_dokument_cos": raw["inhalte_dokument"][i, j],
                "inhalte_stichpunkte_cos": raw["inhalte_stichpunkte"][i, j],
                "inhalte_tfidf_cos": raw["inhalte_tfidf"][i, j],
                "kompetenzen_cos": raw["kompetenzen"][i, j],
                "literatur_dice": raw["literatur"][i, j],
                "anzahl_gemeinsame_literatur": len(shared_lit.get((i, j), [])),
                "gemeinsame_literatur": "; ".join(lit_label(k) for k in shared_lit.get((i, j), [])),
                "aehnlichste_stichpunkte": " || ".join(
                    f"[{s:.2f}] {point_texts[a]} ⟷ {point_texts[b]}" for s, a, b in matches),
            })
    pairs = pd.DataFrame(rows).sort_values("gesamt_z", ascending=False)
    write_csv(pairs.round(4), output_dir / "paare.csv")

    # Top-N je Modul
    both = pd.concat([
        pairs,
        pairs.rename(columns={"modul_a": "modul_b", "name_a": "name_b", "modul_b": "modul_a", "name_b": "name_a"}),
    ])
    top = (both.sort_values(["modul_a", "gesamt_z"], ascending=[True, False])
           .groupby("modul_a", sort=False).head(TOP_N))
    top.insert(2, "rang", top.groupby("modul_a").cumcount() + 1)
    top = top.rename(columns={"modul_a": "modul", "name_a": "modulname",
                              "modul_b": "aehnliches_modul", "name_b": "aehnlicher_modulname"})
    write_csv(top.round(4), output_dir / "top_n.csv")

    return {"pairs": pairs, "n_canon_merged": sum(1 for k, v in canon.items() if k != v)}


def main() -> None:
    res = compute()
    pairs = res["pairs"]
    print(f"{len(pairs)} Paare berechnet → {DEFAULT_OUTPUT}")
    print(f"Literatur: {res['n_canon_merged']} lit_keys unscharf zusammengeführt")
    print("Top-15 ähnlichste Paare (gesamt_z):")
    for _, r in pairs.head(15).iterrows():
        print(f"  {r['gesamt_z']:5.2f}  {r['modul_a']} {r['name_a'][:40]:40s} ⟷ {r['modul_b']} {r['name_b'][:40]}")


if __name__ == "__main__":
    main()
