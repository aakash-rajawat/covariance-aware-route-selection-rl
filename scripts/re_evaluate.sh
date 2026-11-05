#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-smoke}"

[[ "$MODE" == smoke || "$MODE" == full ]] || {
    echo "Usage: smoke|full"
    exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$HOME/Downloads/M7_REPRO_$(date -u +%Y%m%dT%H%M%SZ)_$$"
PY="${M7_PYTHON:-$HOME/belief-aware-robot-navigation/.venv/bin/python}"

mkdir -p "$RUN"
cp -a "$ROOT/src/." "$RUN/"

STATUS=FAILED

finish() {
    trap - EXIT

    printf '{"status":"%s","mode":"%s"}\n' \
        "$STATUS" "$MODE" > "$RUN/STATUS.json"

    python3 - "$RUN" <<'PYZIP'
import sys, zipfile, pathlib

p = pathlib.Path(sys.argv[1])
out = p.with_suffix('.zip')

with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.rglob('*')):
        if f.is_file():
            z.write(f, f.relative_to(p))

with zipfile.ZipFile(out) as z:
    assert z.testzip() is None

print('RESULT_ZIP:', out)
PYZIP
}

trap finish EXIT

[[ -x "$PY" ]] || {
    echo "Missing Python: $PY"
    exit 1
}

"$PY" "$RUN/m7d_replication.py" --mode smoke \
    | tee "$RUN/smoke.log"

if [[ "$MODE" == full ]]; then
    for i in 0 1 2; do
        "$PY" "$RUN/m7d_replication.py" --mode eval \
            --point "$ROOT/checkpoints/seed$i/point.pt" \
            --belief "$ROOT/checkpoints/seed$i/belief.pt" \
            --label "seed$i" \
            | tee "$RUN/eval_seed$i.log"
    done

    "$PY" "$RUN/m7d_replication.py" --mode fixed_baselines \
        | tee "$RUN/baselines.log"
fi

STATUS=PASS_EXECUTION
echo "M7 reproduction PASS_EXECUTION"
