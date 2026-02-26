#!/usr/bin/env bash
# Start maasv HTTP API server with port cleanup
set -euo pipefail

PORT="${MAASV_PORT:-18790}"
PROJECT_DIR="/Users/macmini/Projects/maasv"

cd "$PROJECT_DIR"

# Kill any orphan process on our port
cleanup_port() {
    local pid
    pid=$(lsof -ti "tcp:$PORT" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "[start_server] Port $PORT held by PID $pid — sending SIGTERM"
        kill "$pid" 2>/dev/null || true
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[start_server] PID $pid exited after ${i}s"
                return 0
            fi
            sleep 1
        done
        echo "[start_server] PID $pid still alive — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            echo "[start_server] FATAL: Cannot free port $PORT" >&2
            exit 1
        fi
    fi
}

cleanup_port
exec "$PROJECT_DIR/.venv/bin/maasv-server"
