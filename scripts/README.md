# scripts/ — run tooling

Operational scaffolding for running the experiments unattended. **None of this is
part of the research method**; the analysis lives entirely in `src/`. These files
exist only to launch long jobs, keep the machine awake, recover from crashes, and
consolidate outputs. They are safe to ignore when reading or reviewing the science.

| file | purpose |
|---|---|
| `supervisor.sh` | Orchestrates the primary run (DistilBERT + external validity + significance + mitigation): anti-sleep watchdog, crash/OOM auto-resume from checkpoint, per-task sentinels. |
| `supervisor_ceas.sh` | Same pattern for the CEAS-2008 transformer replication. |
| `finalizer.py` | Polls task sentinels, regenerates figures, writes the consolidated numbers to a standalone run report under `logs/`, runs a consistency pass and the test suite, writes `STATUS.md`. |

Run from the repository root, e.g.:

```bash
bash scripts/supervisor.sh          # primary run, fully detached
python scripts/finalizer.py --now   # consolidate immediately, no sentinel wait
```

Outputs that these tools generate (`logs/`, `results/sentinels/`, the run report)
are git-ignored run state, not committed artifacts.
