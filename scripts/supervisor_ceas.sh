#!/usr/bin/env bash
# supervisor_ceas.sh — autonomous orchestrator for the CEAS-2008 transformer
# experiment, reusing the same pattern as the primary run:
#
#   1. Anti-sleep: caffeinate -dimsu -t 86400, detached, with a
#      5-minute watchdog that respawns it if it dies (the original OOM was caused
#      by caffeinate dying -> Mac slept -> swapped -> jetsam SIGKILL'd training).
#   2. Train DistilBERT on the CEAS-2008 train split + score the cached CEAS Haiku
#      rewrites (NO API calls) via `python -m src.transformer_ceas --all`. Auto-
#      resume from the mid-epoch checkpoint on an OOM death, up to a retry cap.
#   3. On success write the verdict to a standalone run report and append a
#      STATUS section; write results/sentinels/taskE.{done,failed} with a resume
#      command.
#
# Fully detached; survives the controlling shell ending. Idempotent: if the CSV
# already exists it skips training and just (re)consolidates.

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root (this script lives in scripts/)
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
SENT="$ROOT/results/sentinels"
LOGD="$ROOT/logs"
mkdir -p "$SENT" "$LOGD"
SLOG="$LOGD/supervisor_ceas.log"
log(){ echo "[$(date '+%F %T')] $*" >>"$SLOG"; }

log "=== supervisor_ceas start (pid $$) root=$ROOT ==="

# 1. caffeinate: anti-sleep + watchdog
ensure_caffeinate(){
  if ! pgrep -x caffeinate >/dev/null 2>&1; then
    nohup caffeinate -dimsu -t 86400 >/dev/null 2>&1 & disown
    log "caffeinate (re)spawned: caffeinate -dimsu -t 86400"
  fi
}
ensure_caffeinate
( while true; do sleep 300; ensure_caffeinate; done ) & disown
log "caffeinate watchdog started (pid $!) — checks every 5 min"

# sentinel helpers
mark_done(){  # task, message
  local t="$1"; shift
  rm -f "$SENT/$t.failed"
  printf 'DONE: %s\n' "$*" >"$SENT/$t.done"
  log "$t -> DONE: $*"
}
mark_failed(){  # task, reason, resume-cmd
  local t="$1" reason="$2" cmd="$3"
  rm -f "$SENT/$t.done"
  { echo "FAILED: $reason"; echo "RESUME: $cmd"; } >"$SENT/$t.failed"
  log "$t -> FAILED: $reason"
}

# 2. Train CEAS transformer + score cached rewrites (no API)
#    Auto-resume from checkpoint on OOM death, up to MAXR.
E_CSV="$ROOT/results/tables/transformer_ceas_degradation.csv"
RESUME_CMD="PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 $PY -u -m src.transformer_ceas --all"
MAXR=6; retries=0
launch_E(){
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 nohup "$PY" -u -m src.transformer_ceas --all \
    >>"$LOGD/transformer_ceas_run.log" 2>&1 & disown
  log "CEAS transformer run (re)launched (pid $!) — trains+scores; resumes from checkpoint; retries=$retries/$MAXR"
  sleep 20
}
log "CEAS transformer supervisor loop begins"
while true; do
  ensure_caffeinate
  if [ -f "$E_CSV" ]; then
    break
  fi
  if ! pgrep -f "src.transformer_ceas" >/dev/null 2>&1; then
    if [ "$retries" -ge "$MAXR" ]; then
      mark_failed taskE "exceeded $MAXR auto-resumes without producing CSV (persistent OOM?)" \
                  "$RESUME_CMD"
      break
    fi
    retries=$((retries + 1))
    launch_E
  fi
  sleep 60
done

# 3. Consolidate (only if the CSV was produced) — write run report + STATUS
if [ -f "$E_CSV" ]; then
  if "$PY" -m src.transformer_ceas --consolidate >>"$LOGD/transformer_ceas_run.log" 2>&1; then
    # append a STATUS section (idempotent: drop any prior CEAS block first)
    "$PY" - "$ROOT" >>"$LOGD/transformer_ceas_run.log" 2>&1 <<'PYEOF'
import sys, datetime, pandas as pd
from pathlib import Path
root = Path(sys.argv[1])
deg = pd.read_csv(root/"results/tables/transformer_ceas_degradation.csv")
mcn = pd.read_csv(root/"results/tables/transformer_ceas_significance.csv")
hb = deg[(deg["set"]=="degradation") & (deg["generator"]=="haiku")]
def drop(cond):
    s = hb[hb.condition==cond].set_index("severity")
    return (s.loc[1.0,"recall_05"]-s.loc[0.0,"recall_05"])*100
do, dm = drop("original"), drop("url_masked")
mo = mcn[(mcn.generator=="haiku")&(mcn.condition=="original")].iloc[0]
mm = mcn[(mcn.generator=="haiku")&(mcn.condition=="url_masked")].iloc[0]
n = int(hb["n_phish"].iloc[0])
BEGIN="<!-- STATUS:ceas-transformer BEGIN -->"; END="<!-- STATUS:ceas-transformer END -->"
block = f"""{BEGIN}
## CEAS-2008 transformer replication

_Updated {datetime.datetime.now():%Y-%m-%d %H:%M} by supervisor_ceas.sh_

- **DONE**: `results/tables/transformer_ceas_degradation.csv` + `transformer_ceas_significance.csv` (no API calls).
- DistilBERT fine-tuned on CEAS-2008 train split; scored on cached CEAS Haiku rewrites; intersection n={n}.
- Text-robustness: recall@0.5 sev0->sev1 drop {do:+.1f} pts (McNemar p={mo['mcnemar_exact_p']:.2g}).
- URL-anchoring: url-masked sev0->sev1 drop {dm:+.1f} pts (McNemar p={mm['mcnemar_exact_p']:.2g}).
- Verdict written to logs/ceas_transformer_report.md.
{END}"""
sp = root/"STATUS.md"
t = sp.read_text() if sp.exists() else ""
if BEGIN in t and END in t:
    t = t[:t.index(BEGIN)] + block + t[t.index(END)+len(END):]
else:
    t = t.rstrip() + "\n\n" + block + "\n"
sp.write_text(t)
print(f"STATUS.md updated (CEAS transformer, n={n}, drop_o={do:+.1f}, drop_m={dm:+.1f})")
PYEOF
    mark_done taskE "transformer_ceas_degradation.csv produced; written to run report + STATUS.md"
  else
    mark_failed taskE "scoring CSV present but consolidation failed" \
                "$PY -m src.transformer_ceas --consolidate"
  fi
fi

log "=== supervisor_ceas: task settled; exiting (caffeinate watchdog continues) ==="
