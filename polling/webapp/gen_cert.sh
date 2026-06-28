#!/usr/bin/env bash
# Generate a self-signed certificate so the Coach can be served over HTTPS.
#
# Why: browsers only allow camera access (getUserMedia) in a "secure context".
# Over a LAN IP that means HTTPS — plain http://<jetson-ip>:8000 will NOT let a
# phone use its camera. This cert enables the "This phone / laptop" camera source.
# It's self-signed, so the phone shows a one-time "Not private" warning the first
# time — tap Advanced -> Proceed. (The Jetson-camera mode works fine over HTTP.)
set -euo pipefail
cd "$(dirname "$0")"   # polling/webapp/

openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout key.pem -out cert.pem \
  -subj "/CN=mvgpt-coach" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null

echo "Wrote cert.pem + key.pem in $(pwd)"
echo "Now (re)start with:  bash polling/webapp/run.sh   (it auto-detects the cert and serves HTTPS)"
