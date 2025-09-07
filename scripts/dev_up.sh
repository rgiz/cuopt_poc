#!/usr/bin/env bash
set -euo pipefail

# --- config ---
UI_HOST_PORT="${UI_HOST_PORT:-8501}"   # override with: UI_HOST_PORT=8501 ./scripts/dev_up.sh

echo "▶ Ensuring host port ${UI_HOST_PORT} is free…"

# A) stop any *docker container* publishing ${UI_HOST_PORT}
#    (works cross-platform; doesn’t rely on lsof)
mapfile -t DOCKER_PORT_USERS < <(
  docker ps --format '{{.ID}} {{.Names}} {{.Ports}}' \
  | awk -v p=":${UI_HOST_PORT}->" 'index($0,p){print $1}'
)
if ((${#DOCKER_PORT_USERS[@]})); then
  echo "→ Stopping containers using port ${UI_HOST_PORT}: ${DOCKER_PORT_USERS[*]}"
  docker stop "${DOCKER_PORT_USERS[@]}" >/dev/null
fi

# B) also kill any *host process* listening on ${UI_HOST_PORT}
if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:${UI_HOST_PORT} -sTCP:LISTEN -P >/dev/null 2>&1; then
    echo "→ Killing host processes on ${UI_HOST_PORT}"
    # shellcheck disable=SC2046
    kill -9 $(lsof -t -iTCP:${UI_HOST_PORT} -sTCP:LISTEN) || true
  fi
else
  echo "ℹ lsof not found — skipping host process kill step"
fi

echo "▶ Bringing stack down (clean)…"
docker compose down --remove-orphans

echo "▶ Optional light cleanup…"
# remove exited containers and dangling images only (safe)
docker container prune -f >/dev/null || true
docker image prune -f >/dev/null || true
docker network prune -f >/dev/null || true

echo "▶ Building images…"
docker compose build

echo "▶ Starting services…"
docker compose up -d

echo "▶ Status:"
docker compose ps

# print a friendly URL for the UI
UI_MAPPED="$(docker compose ps ui --format json \
  | jq -r '.[0].Publishers[]? | select(.TargetPort==8501) | "\(.Url)"' 2>/dev/null || true)"

if [[ -n "${UI_MAPPED}" ]]; then
  echo "✅ UI available at: ${UI_MAPPED}"
else
  echo "✅ UI should be on http://localhost:${UI_HOST_PORT} (if you mapped it explicitly)"
fi

echo "▶ Health check:"
if command -v curl >/dev/null 2>&1; then
  curl -fsS http://localhost:8000/health || true
else
  echo "  (curl not installed; skipping)"
fi
