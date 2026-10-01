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
ausgeblendete Rahmenmodule und Größengrenzen in `export_web.py` (`RAHMENMODULE`, `CAPS`).

## Website lokal ansehen

```bash
cd web && python3 -m http.server 8765   # http://localhost:8765
```

## Deployment (Sliplane)

Das `Dockerfile` im Projektwurzelverzeichnis baut einen nginx-Container mit der statischen Website (Port 80,
Healthcheck unter `/healthz`). In Sliplane einen Service aus dem Git-Repository mit diesem Dockerfile anlegen.
Nach Datenänderungen die Pipeline neu laufen lassen, `web/data/data.json` committen und neu deployen.
