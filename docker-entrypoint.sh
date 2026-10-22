#!/bin/sh
set -eu

mkdir -p "$HF_HOME" "$CLASSLENS_MODEL_CACHE"
chown classlens:classlens /var/data "$HF_HOME" "$CLASSLENS_MODEL_CACHE"
if [ "${1:-}" = "uvicorn" ]; then
    port="${PORT:-10000}"
    case "$port" in
        ''|*[!0-9]*)
            echo "PORT must be a numeric TCP port." >&2
            exit 1
            ;;
    esac
    if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        echo "PORT must be between 1 and 65535." >&2
        exit 1
    fi
    set -- "$@" --port "$port"
fi
exec su -p -s /bin/sh classlens -c "exec $*"
