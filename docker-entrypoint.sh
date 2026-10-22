#!/bin/sh
set -eu

mkdir -p "$HF_HOME" "$CLASSLENS_MODEL_CACHE"
chown classlens:classlens /var/data "$HF_HOME" "$CLASSLENS_MODEL_CACHE"
exec su -p -s /bin/sh classlens -c "exec $*"
