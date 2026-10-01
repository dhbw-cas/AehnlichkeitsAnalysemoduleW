# Statische Website „Modullandschaft“ – alle Berechnungen sind vorab in web/data/data.json exportiert.
FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY web/ /usr/share/nginx/html/
EXPOSE 80
