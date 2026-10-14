#!/usr/bin/env bash
# Run the port-equivalence checks (static AST check + bit-for-bit numerical tests) once the machine is free.
#
# The numerical tests use the GPU and many CPU threads, so they must not overlap a long run. This script waits until
# the run declares itself finished (a marker file exists AND its systemd unit is no longer active), then:
#   1. checks every ported module against its original with scripts/check_port_ast.py (syntax trees);
#   2. runs tests/test_kernels_equivalence.py (CPU fallbacks and Triton kernels, bit-identical outputs).
#
# Machine-specific paths are read from scripts/equivalence_paths.env (not versioned; copy
# scripts/equivalence_paths.env.example and fill it in) or from the environment.
#
# Usage: scripts/run_equivalence_after_passata.sh [--no-wait] [extra pytest arguments]
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ENV_FILE="${FUCINA_EQUIVALENCE_ENV:-$HERE/equivalence_paths.env}"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && . "$ENV_FILE"

WAIT=1
if [ "${1:-}" = "--no-wait" ]; then WAIT=0; shift; fi

need() { [ -n "${!1:-}" ] || { echo "missing setting: $1 (see $HERE/equivalence_paths.env.example)" >&2; exit 2; }; }
for v in FUCINA_ORIG_THOR_V2 FUCINA_ORIG_S23 FUCINA_ORIG_KERNEL_ROT FUCINA_GGUF_PY PYTHON; do need "$v"; done
export FUCINA_ORIG_THOR_V2 FUCINA_ORIG_S23 FUCINA_ORIG_KERNEL_ROT FUCINA_GGUF_PY
export FUCINA_RD_TERNARY_DIR="${FUCINA_RD_TERNARY_DIR:-$FUCINA_ORIG_S23/lib}"
export FUCINA_REFINE_DIR="${FUCINA_REFINE_DIR:-$FUCINA_ORIG_S23/rif}"

# ── 0. wait for the running job ────────────────────────────────────────────────────────────────────────────────
if [ "$WAIT" = 1 ]; then
    need WAIT_MARKER
    UNIT="${WAIT_UNIT:-}"
    echo "$(date '+%F %T') waiting for $WAIT_MARKER${UNIT:+ and for unit $UNIT to stop}"
    while :; do
        busy=0
        [ -e "$WAIT_MARKER" ] || busy=1
        if [ -n "$UNIT" ] && systemctl --user is-active --quiet "$UNIT" 2>/dev/null; then busy=1; fi
        [ "$busy" = 0 ] && break
        sleep 120
    done
    echo "$(date '+%F %T') the run is finished; starting the checks"
fi

# ── pytest: from $PYTHON, or (pure-Python packages only) from FUCINA_PYTEST_SITE ───────────────────────────────
PYPATH="$ROOT"
if ! "$PYTHON" -c "import pytest" 2>/dev/null; then
    if [ -n "${FUCINA_PYTEST_SITE:-}" ]; then
        SHIM="$ROOT/.equivalence_pytest"
        mkdir -p "$SHIM"
        for pkg in _pytest pytest pluggy iniconfig packaging py.py pygments; do
            [ -e "$FUCINA_PYTEST_SITE/$pkg" ] && ln -sfn "$FUCINA_PYTEST_SITE/$pkg" "$SHIM/$pkg"
        done
        PYPATH="$ROOT:$SHIM"
    else
        echo "pytest is not installed for $PYTHON: install it ($PYTHON -m pip install pytest) or set FUCINA_PYTEST_SITE" >&2
        exit 2
    fi
fi

LOG_DIR="${EQUIVALENCE_LOG_DIR:-$ROOT/equivalence-logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ── 1. static check ────────────────────────────────────────────────────────────────────────────────────────────
echo "== static AST check" | tee "$LOG_DIR/ast_$STAMP.log"
ast_fail=0
while IFS=$'\t' read -r orig_mod port_mod; do
    case "$orig_mod" in ''|'#'*) continue ;; esac
    rel="${orig_mod//.//}.py"
    src=""
    for base in "$FUCINA_ORIG_THOR_V2" "$FUCINA_ORIG_S23/s23" "$FUCINA_ORIG_S23/lib" "$FUCINA_ORIG_S23/s1b" \
                "$FUCINA_ORIG_S23/rif" "$FUCINA_ORIG_KERNEL_ROT"; do
        [ -f "$base/$rel" ] && { src="$base/$rel"; break; }
    done
    dst="$ROOT/${port_mod//.//}.py"
    if [ -z "$src" ] || [ ! -f "$dst" ]; then
        echo "SKIPPED    $orig_mod -> $port_mod (not ported yet or original not found)" | tee -a "$LOG_DIR/ast_$STAMP.log"
        continue
    fi
    "$PYTHON" "$ROOT/scripts/check_port_ast.py" --map "$ROOT/docs/port_identifier_map.tsv" \
        --modules "$ROOT/docs/port_module_map.tsv" --external "$ROOT/docs/port_external_names.tsv" \
        --expected-dir "$ROOT/tests/port_baseline" "$src" "$dst" --max-lines 60 >> "$LOG_DIR/ast_$STAMP.log" 2>&1 \
        || ast_fail=1
done < "$ROOT/docs/port_module_map.tsv"
grep -E "^(IDENTICAL|DIFFERENT|SKIPPED)" "$LOG_DIR/ast_$STAMP.log"

# ── 2. numerical equivalence ───────────────────────────────────────────────────────────────────────────────────
echo "== numerical equivalence (log: $LOG_DIR/pytest_$STAMP.log)"
cd "$ROOT" || exit 2
PYTHONPATH="$PYPATH${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -m pytest -p no:cacheprovider -rsxX tests/test_kernels_equivalence.py "$@" 2>&1 \
    | tee "$LOG_DIR/pytest_$STAMP.log"
py_status=${PIPESTATUS[0]}

echo "== summary: AST $([ $ast_fail = 0 ] && echo OK || echo DIFFERENCES) | pytest exit $py_status"
[ "$ast_fail" = 0 ] && [ "$py_status" = 0 ]
