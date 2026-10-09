# Ähnlichkeitsanalyse Module Wirtschaft

Vergleicht die Module des Masterprogramms Wirtschaft paarweise nach Inhalten, Kompetenzen und Literatur
und zeigt die Clusterung als interaktive Website („Modullandschaft“).

## Pipeline

```bash
uv run python -m 1082_aehnlichkeitsanalyse_module_wirtschaft.preprocess   # data/module.csv → data/aufbereitet/
uv run python -m 1082_aehnlichkeitsanalyse_module_wirtschaft.similarity   # → data/aehnlichkeit/ (erster Lauf ~10 min, danach Cache)
uv run python -m 1082_aehnlichkeitsanalyse_module_wirtschaft.export_web   # → web/data/data.json
```

Gewichte und Parameter stehen oben in `similarity.py` (`WEIGHTS`, `CONTENT_MIX`, `LIT_SCALE`);
ausgeblendete Rahmenmodule, Größengrenzen und die Zuordnung Modulnummer → Studiengang in `export_web.py`
(`RAHMENMODULE`, `CAPS`, `STUDIENGAENGE`).

## Website lokal ansehen

```bash
cd web && python3 -m http.server 8765   # http://localhost:8765
```

## Links und Export

- Einstellungen stehen im Link: `#k=90&max=5` (90 Cluster, höchstens 5 Module je Cluster).
- Gewichtung: Anwender können Inhalte, Kompetenzen und Literatur im Bereich „Einstellungen“ selbst gewichten.
  Eine abweichende Gewichtung steht als `&gewichte=60-20-20` (Inhalte-Kompetenzen-Literatur) im Link; die
  Website berechnet Gesamtwert und Clusterung dann im Browser neu (gleicher Algorithmus wie `export_web.py`).
  Die Karte behält ihre Anordnung aus der Standardgewichtung (`WEIGHTS`). Ohne `gewichte` gelten die
  vorab berechneten Werte – bestehende Links bleiben unverändert.
- Feste Ansicht ohne Regler, z. B. für Arbeitsgruppen: `#k=90&max=5&ansicht=fest`
  (Button „Link zur festen Ansicht“). Das blendet die Regler nur aus und ist kein Zugriffsschutz.
- „Excel-Export“ lädt den aktuellen Stand als `.xlsx` (Blätter Cluster, Module, Hinweise) mit leeren
  Spalten „Bewertung“ und „Kommentar“. Cluster-Nummern gelten nur für die jeweilige Einstellung.

## Deployment (Sliplane)

Das `Dockerfile` im Projektwurzelverzeichnis baut einen nginx-Container mit der statischen Website (Port 80,
Healthcheck unter `/healthz`). In Sliplane einen Service aus dem Git-Repository mit diesem Dockerfile anlegen.
Nach Datenänderungen die Pipeline neu laufen lassen, `web/data/data.json` committen und neu deployen.
