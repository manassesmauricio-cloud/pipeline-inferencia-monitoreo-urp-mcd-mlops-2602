#!/usr/bin/env bash
# Ensayo local: ejecuta exactamente lo mismo que el runner autohospedado (CI y, si pasa, CD), sin pasar por GitHub.
set -euo pipefail
cd "$(dirname "$0")"
bash ci_local.sh
bash cd_local.sh
