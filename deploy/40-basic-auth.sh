#!/bin/sh
# Passwortschutz: schreibt beim Containerstart die Zugangsdaten für nginx (auth_basic).
# BASIC_AUTH_PASSWORD ist Pflicht – ohne Passwort startet der Container nicht, damit die Seite nie
# versehentlich offen ist. BASIC_AUTH_USER ist optional (Standard: modullandschaft).
set -eu

user="${BASIC_AUTH_USER:-modullandschaft}"
if [ -z "${BASIC_AUTH_PASSWORD:-}" ]; then
    echo "$0: BASIC_AUTH_PASSWORD ist nicht gesetzt – Passwortschutz nicht möglich, Abbruch" >&2
    exit 1
fi
case "$user" in
    *:*) echo "$0: BASIC_AUTH_USER darf keinen Doppelpunkt enthalten" >&2; exit 1 ;;
esac

# Gesalzener SHA-512-Hash; das Klartextpasswort landet in keiner Datei
hash=$(printf '%s\n' "$BASIC_AUTH_PASSWORD" | openssl passwd -6 -stdin)
printf '%s:%s\n' "$user" "$hash" > /etc/nginx/htpasswd
chown root:nginx /etc/nginx/htpasswd
chmod 640 /etc/nginx/htpasswd
echo "$0: Passwortschutz aktiv für Benutzer „$user“"
