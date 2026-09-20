# Pilot evidence

- **`power-model-pilot-verified/` is authoritative:** the completed 29-case repeated suite, with correct process-tree memory measurements. See `summary.json`, `manifest.json`, and per-run compressed traces.
- `power-model-pilot/` is the first complete suite. Its trajectories match the verified suite, but its memory readings cover only the Windows launcher and must not be used as workload RAM figures.
- `power-model-pilot-startup-check/` preserves the initial output-path failure. No complete exported simulation result is present there.

The repeated cases use the same seeds; they demonstrate numerical reproducibility, not independent statistical replication. The [pilot report](../reports/2026-09-20-power-model-pilot.md) explains scope, results, limitations, and reproduction.
