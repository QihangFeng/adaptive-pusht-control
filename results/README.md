# Pilot evidence

- **`power-model-pilot-verified/` is authoritative:** the completed 29-case repeated suite, with correct process-tree memory measurements. See `summary.json`, `manifest.json`, and per-run compressed traces.
- `power-model-pilot/` is the first complete suite. Its trajectories match the verified suite, but its memory readings cover only the Windows launcher and must not be used as workload RAM figures.
- `power-model-pilot-startup-check/` preserves the initial output-path failure. No complete exported simulation result is present there.

The repeated cases use the same seeds; they demonstrate numerical reproducibility, not independent statistical replication. The [pilot report](../reports/2026-09-20-power-model-pilot.md) explains scope, results, limitations, and reproduction.

## Delayed-voltage topic screening

- `delay-screen/`: 30 baseline cases, predeclared protocol, exact zero-delay compatibility check, and development-only tuning selection.
- `delay-prototype/`: 18 cases using the frozen selected setting; includes the minimal power-history predictor and two evaluation schedules with two seeds.
- [Gap audit and decision report](../reports/2026-09-20-delay-go-no-go.md): interpretation, literature, limitations, CPU resource use, and reproduction commands. These results support a course-project extension under stated assumptions, not deployment safety or general algorithmic novelty.

## Absolute acceptability audit

- `delay-acceptability-audit/`: a standard-library reanalysis of the same 48 saved runs, not new simulations. Includes source/trace hashes, unchanged experimental budgets, per-run failures, and paired comparisons.
- [Acceptability report](../reports/2026-10-08-delay-acceptability-audit.md): only one of four corrected evaluation cases meets all earlier experimental budgets; none of the 48 runs meets the strict nominal-voltage check. The environment was missing and slower-update evaluation was pending at audit time; see the completed follow-up below. Independent grid-disturbance tests remain pending.

## Restored environment and slower-controller comparison

- `simulator-restoration/`: provenance for the recreated numerical inputs and runtime. Numerical files match the historical hashes; the new manifest's only difference is its truthful creation timestamp.
- `slow-update-check/`: two exact historical-trajectory checks followed by four new frozen `slow_update` cases. Includes the protocol, traces, absolute-budget checks and comparison against saved correction runs.
- [Comparison report](../reports/2026-10-08-slow-update-comparison.md): slower stale control passes zero of four cases; correction passes one of four. Relative gains do not imply reliable absolute compliance.
- [Persistent setup and reproduction](../reports/2026-10-08-simulator-restoration.md): replaces the earlier temporary-directory setup instructions.
