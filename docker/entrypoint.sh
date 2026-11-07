#!/bin/sh
# Container entrypoint: "api" (default) runs uvicorn, "worker" runs the huey consumer.
set -eu

# Florence weights come from FLORENCE_HF_REPO (or a mounted volume). Failure is reported by /readyz.
python scripts/fetch_models.py florence || echo "Florence adapter not downloaded; see /readyz"

case "${1:-api}" in
  api)
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips="*"
    ;;
  worker)
    exec huey_consumer api.worker.huey --workers "${WORKER_CONCURRENCY:-1}"
    ;;
  *)
    exec "$@"
    ;;
esac
