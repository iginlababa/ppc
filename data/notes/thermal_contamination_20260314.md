# Thermal Contamination Notice — 2026-03-14

## Affected Files

| File | Rows affected | Reason |
|------|---------------|--------|
| `data/raw/quarantine/stream_raja_nvidia_rtx5060_20260314.csv` | all 30 rows | Full session thermally throttled |
| `data/raw/stream_native_nvidia_rtx5060_20260314.csv` | rows with `timestamp LIKE '2026-03-14T16:52%'` (rows 61–90, run_id 1–30 from second append) | GPU throttled at session start |

## What happened

After a prolonged RAJA build session, the RTX 5060 GPU entered sustained
thermal throttle before and during the 16:52 run.

### RAJA CSV (all 30 rows bad)
- Runs 1–9 (hw_state=0, flagged as outliers by the script): ~244–269 GB/s — GPU
  was briefly in boost state but already warming up.
- Runs 10–30 (hw_state=1, script considered "clean"): ~172–173 GB/s — GPU had
  fully throttled to its sustained power limit.
- The script's outlier detection correctly identified the 9 boost runs as
  outliers relative to the 21-run throttled cluster, but the throttled cluster
  is the scientifically invalid result.
- Reported median 172.5 GB/s → PPC ≈ 0.49 vs clean native baseline (meaningless).

### Native CSV (rows 61–90 bad, 00:38 session rows 1–60 are clean)
- The 00:38 session (rows 1–60 in the CSV) recorded 350 GB/s — full boost state.
- The 16:52 session (rows 61–90, appended later) recorded 229–255 GB/s — GPU
  throttled before the run even started.

## Action taken

- `stream_raja_nvidia_rtx5060_20260314.csv` → moved to `quarantine/`.
- Native CSV left in place (rows 1–60 are clean); the 16:52 rows are documented
  here and must be excluded in analysis by filtering `timestamp < '2026-03-14T16:00:00Z'`.
- All 5 abstractions rerun in a single thermally-controlled session on 2026-03-14
  with 5-minute GPU cooldown between abstractions to ensure comparable
  boost-state conditions. The quarantined RAJA file freed the filename, so the
  fresh rerun uses the standard `stream_raja_nvidia_rtx5060_20260314.csv`.

## Exclusion filter (for analysis scripts)

```sql
-- Exclude contaminated rows from native CSV
WHERE abstraction = 'native'
  AND timestamp < '2026-03-14T16:00:00Z'

-- Exclude entire original raja CSV (quarantined in data/raw/quarantine/)
-- Use: stream_raja_nvidia_rtx5060_20260314.csv (fresh rerun)
```

## Controlled 5-abstraction rerun results (5-min cooldown between each)

Run order: native → kokkos → raja → (sycl skipped) → julia → numba

| Abstraction | Median GB/s | Clean | Thermal state observed |
|-------------|------------|-------|------------------------|
| native | 277.1 | 22/30 | sustained (~0.79 × peak) |
| kokkos | 350.1 | 21/30 | **boost** — artifact |
| raja | 274.4 | 23/30 | sustained |
| julia | 270.7 | 23/30 | sustained |
| numba | 349.7 | 30/30 | **boost** — artifact |

Even with 5-min cooldowns the RTX 5060 GPU non-deterministically
entered boost state for kokkos and numba while staying at sustained clocks
for native, RAJA, and julia. Kokkos/numba "outperforming" native is a
measurement artifact from laptop clock non-determinism, not a real speed-up.

**RAJA interpretation:** native and RAJA both ran at sustained clocks in
the same session. PPC = 274.4 / 277.1 = **0.990** — overhead-free,
"excellent" per §9.4 thresholds. This is the valid PPC figure.

**Analysis recommendation:** for cross-abstraction comparison, use only
runs where all abstractions share the same thermal regime. Cite peak
hardware bandwidth from the 00:38 native session (350 GB/s) as the
roofline ceiling, and each abstraction's sustained-state median for PPC.

## Clean reference

| Abstraction | Clean file | Session | Notes |
|-------------|-----------|---------|-------|
| native (peak) | `stream_native_nvidia_rtx5060_20260314.csv` rows 1–60 (ts 00:38) | 00:38 | 350 GB/s — true hardware peak |
| native (sustained) | same file, rows 91–120 (ts ~18:xx) | controlled rerun | 277 GB/s |
| kokkos | `stream_kokkos_nvidia_rtx5060_20260314.csv` latest 30 rows | controlled rerun | 350 GB/s — boost state (artifact) |
| julia | `stream_julia_nvidia_rtx5060_20260314.csv` latest 30 rows | controlled rerun | 271 GB/s — sustained |
| numba | `stream_numba_nvidia_rtx5060_20260314.csv` latest 30 rows | controlled rerun | 350 GB/s — boost state (artifact) |
| raja | `stream_raja_nvidia_rtx5060_20260314.csv` | controlled rerun | 274 GB/s — sustained, **PPC=0.990** |
