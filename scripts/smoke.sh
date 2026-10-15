#!/usr/bin/env bash
# Transcribe a file through the running server, without any client library.
#
#   ./scripts/smoke.sh                     # uses tests/audio/hello.wav
#   ./scripts/smoke.sh sample.wav
#   URL=http://gpu-node:8000 ./scripts/smoke.sh sample.wav
#
# Also checks /health and /v1/models first, so a failure points at the right
# layer: model loading, routing, or the request itself.
set -euo pipefail

URL="${URL:-http://127.0.0.1:8000}"
AUDIO="${1:-tests/audio/hello.wav}"
API_KEY="${API_KEY:-}"

auth=()
if [[ -n "${API_KEY}" ]]; then
    auth=(-H "Authorization: Bearer ${API_KEY}")
fi

if [[ ! -f "${AUDIO}" ]]; then
    echo "no such audio file: ${AUDIO}" >&2
    echo "generate one with: python scripts/make_sample_audio.py ${AUDIO}" >&2
    exit 1
fi

echo "==> GET ${URL}/health"
curl -fsS "${URL}/health"
echo

echo "==> GET ${URL}/v1/models"
curl -fsS "${auth[@]}" "${URL}/v1/models"
echo

echo "==> POST ${URL}/v1/audio/transcriptions (${AUDIO})"
curl -fsS "${auth[@]}" \
    -F "file=@${AUDIO}" \
    -F "response_format=verbose_json" \
    "${URL}/v1/audio/transcriptions"
echo

echo "==> POST with stream=true"
curl -fsS -N "${auth[@]}" \
    -F "file=@${AUDIO}" \
    -F "stream=true" \
    "${URL}/v1/audio/transcriptions" | head -20
