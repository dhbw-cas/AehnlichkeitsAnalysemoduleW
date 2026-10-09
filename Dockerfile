# Statische Website „Modullandschaft“ – alle Berechnungen sind vorab in web/data/data.json exportiert.
FROM nginx:1.27-alpine
# openssl erzeugt beim Start den Passwort-Hash (deploy/40-basic-auth.sh)
RUN apk add --no-cache openssl
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/40-basic-auth.sh /docker-entrypoint.d/40-basic-auth.sh
RUN chmod 755 /docker-entrypoint.d/40-basic-auth.sh
COPY web/ /usr/share/nginx/html/
EXPOSE 80
