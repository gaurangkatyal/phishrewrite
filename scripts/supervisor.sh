#!/usr/bin/env bash
# supervisor.sh — autonomous overnight orchestrator for the four follow-up runs
# (transformer, external validity, paired significance, reverse mitigation).
#
# Responsibilities:
#   1. Own anti-sleep for the whole window: caffeinate -dimsu -t 86400, detached,
#      with a 5-minute watchdog that respawns it if it dies (the original OOM was
#      caused by caffeinate dying -> Mac slept -> swapped -> jetsam killed train).
#   2. Settle the external-validity, significance, and reverse-mitigation runs:
#      write a sentinel if already complete, else run them resuming from cache; on
#      failure preserve cache and write a .failed sentinel with the exact resume
#      command (no crash-loop).
#   3. Adopt the already-running transformer job; auto-resume it from its mid-epoch
#      checkpoint if it OOM-dies, up to a retry cap; write taskA.done/.failed.
#   4. Launch the finalizer, which waits on all sentinels and writes STATUS.md.
#
# Fully detached; survives the controlling shell/session ending. Idempotent.

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root (this script lives in scripts/)
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
SENT="$ROOT/results/sentinels"
LOGD="$ROOT/logs"
mkdir -p "$SENT" "$LOGD"
SLOG="$LOGD/supervisor.log"
log(){ echo "[$(date '+%F %T')] $*" >>"$SLOG"; }

log "=== supervisor start (pid $$) root=$ROOT ==="

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

# 2. External validity, significance, reverse mitigation — settle from cache
#    (no re-spend if outputs already exist)
# Paired significance — already complete in this repo.
if [ -f "$ROOT/results/tables/significance_paired.csv" ]; then
  mark_done taskC "significance_paired.csv present (paired McNemar exact)"
else
  if "$PY" -m src.significance >>"$LOGD/significance_resume.log" 2>&1 \
     && [ -f "$ROOT/results/tables/significance_paired.csv" ]; then
    mark_done taskC "significance_paired.csv produced"
  else
    mark_failed taskC "significance run did not produce CSV" "$PY -m src.significance"
  fi
fi

# CEAS external validity (~$1.10 Haiku, pre-approved). Resume from cache.
if [ -f "$ROOT/results/tables/external_validity.csv" ]; then
  mark_done taskB "external_validity.csv present (CEAS-2008 replication, n=61)"
else
  log "external validity incomplete — running --rewrite --score (resumes from JSONL cache)"
  if "$PY" -m src.external_validity --rewrite --score >>"$LOGD/external_validity_resume.log" 2>&1 \
     && [ -f "$ROOT/results/tables/external_validity.csv" ]; then
    mark_done taskB "external_validity.csv produced on resume"
  else
    mark_failed taskB "rewrite/score failed (check credit/quota; cache preserved)" \
                "$PY -m src.external_validity --rewrite --score"
  fi
fi

# Reverse mitigation: Gemini->Haiku (~$0.91 Gemini, pre-approved).
if [ -f "$ROOT/results/tables/mitigation_cross_haiku.csv" ]; then
  mark_done taskD "mitigation_cross_haiku.csv present (reverse mitigation)"
else
  log "reverse mitigation incomplete — running --run --yes (resumes from JSONL cache)"
  if "$PY" -m src.mitigate_reverse --run --yes >>"$LOGD/mitigate_reverse_resume.log" 2>&1 \
     && [ -f "$ROOT/results/tables/mitigation_cross_haiku.csv" ]; then
    mark_done taskD "mitigation_cross_haiku.csv produced on resume"
  else
    mark_failed taskD "run failed (check credit/quota; cache preserved)" \
                "$PY -m src.mitigate_reverse --run --yes"
  fi
fi

# 4. Launch finalizer (it waits on all sentinels, 12h cap, then writes STATUS.md)
if ! pgrep -f "scripts/finalizer.py" >/dev/null 2>&1; then
  nohup "$PY" "$ROOT/scripts/finalizer.py" >>"$LOGD/finalizer.log" 2>&1 & disown
  log "finalizer launched (pid $!)"
else
  log "finalizer already running — not relaunching"
fi

# 3. Transformer run — adopt the running job; auto-resume from checkpoint on OOM
A_DEG="$ROOT/results/tables/transformer_degradation.csv"
A_ERA="$ROOT/results/tables/transformer_era.csv"
MAXR=6; retries=0
launch_A(){
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 nohup "$PY" -u -m src.transformer_detector --all \
    >>"$LOGD/transformer_run.log" 2>&1 & disown
  log "transformer run (re)launched (pid $!) — resumes from checkpoint; retries=$retries/$MAXR"
  sleep 20
}
log "transformer supervisor loop begins (adopting existing run if present)"
while true; do
  ensure_caffeinate
  if [ -f "$A_DEG" ] && [ -f "$A_ERA" ]; then
    mark_done taskA "transformer_degradation.csv + transformer_era.csv present"
    break
  fi
  if ! pgrep -f "src.transformer_detector" >/dev/null 2>&1; then
    if [ "$retries" -ge "$MAXR" ]; then
      mark_failed taskA "exceeded $MAXR auto-resumes without producing CSVs (persistent OOM?)" \
                  "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 $PY -u -m src.transformer_detector --all"
      break
    fi
    retries=$((retries + 1))
    launch_A
  fi
  sleep 60
done

log "=== supervisor: all tasks settled; exiting (finalizer continues) ==="
