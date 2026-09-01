#!/usr/bin/env bash
set -euo pipefail

image_name="${1:-neurobridge-s4:ci}"
container_name="neurobridge-s4-smoke-${GITHUB_RUN_ID:-local}-$$"

cleanup() {
  docker rm --force "${container_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker run --detach \
  --name "${container_name}" \
  --publish 127.0.0.1:8501:8501 \
  "${image_name}" >/dev/null

for attempt in {1..30}; do
  if curl --fail --silent --show-error \
    http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "Container health check passed."
    exit 0
  fi

  if ! docker inspect --format '{{.State.Running}}' "${container_name}" | grep -q true; then
    echo "Container exited before becoming healthy." >&2
    docker logs "${container_name}" >&2
    exit 1
  fi

  sleep 2
done

echo "Container did not become healthy within 60 seconds." >&2
docker logs "${container_name}" >&2
exit 1
