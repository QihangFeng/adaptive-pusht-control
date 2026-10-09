# Simulator restoration and persistent local setup

Date: October 8, 2026

## Result

**The simulator is working again in the project's Python 3.11.9 environment.** The numerical inputs match the original files byte-for-byte, and two complete simulation trajectories exactly reproduce saved results. The four missing slower-controller comparisons also completed; see the [comparison report](2026-10-08-slow-update-comparison.md).

No GPU inference or model training was required. The GTX 1650 was not used.

## What was restored

| Component | Persistent project location | Verification |
|---|---|---|
| Python runtime | `.venv/` | Existing Python 3.11.9, with the recorded package versions installed |
| OpenG2G source | `external/openg2g/` | Clean checkout at `c582a5115cfd49dcdfb140f1837224e7cbf0cfb1` |
| Selected ML.ENERGY records | `data/raw/huggingface/` | Pinned dataset revision; seven result files plus the summary parquet |
| Prepared simulation inputs | `data/prepared/b473220852891f26/` | All three numerical artifacts match the original SHA-256 hashes |
| Restoration evidence | [verification.json](../results/simulator-restoration/verification.json) | Revisions, package versions, file hashes, and timestamp comparison |

The raw selection is Llama-3.1-8B-Instruct, H100, one GPU, `lm-arena-chat`, and batches 8, 16, 32, 64, 128, 256, 512, from `ml-energy/benchmark-v3` revision `fea6cb99d037d11f86f9c45a6991b9e5764f4529`. The downloaded files total **170,495,307 bytes**, approximately 170.5 MB, including the summary parquet. This is not a full benchmark download.

The simulator checkout, environment, and raw/prepared datasets are ignored by Git. The restoration scripts, dependency pins, reports, and small verification/results artifacts remain shareable. No authentication token was embedded in code or results; the download used the existing local Hugging Face login.

## What “matching” means

- `trace.csv`, `itl_fit.json`, and `logistic_fit.json` exactly match their historical hashes.
- The new `_manifest.json` truthfully records its new creation time. Replacing only that timestamp **in memory** produces the original manifest hash. The on-disk timestamp was not backdated.
- The recorded package versions match the earlier inventory. Python itself changed from the old temporary environment's 3.11.16 to the user's permanent installation's 3.11.9. This is not an identical entire environment.
- To check the consequence of that runtime change, the new suite reproduced one fresh-feedback trajectory and one two-second-delayed trajectory. Both 3,000-step trajectory hashes match exactly. This verifies those cases, not every possible experiment on every machine.
- The source checkout and all prepared inputs remained unchanged during the six new runs. All six saved runs report converged power flows; the trace audit recomputes their metrics and checks causal observations.

OpenG2G emitted its existing warning that the throughput fit has a negative asymptotic baseline parameter during regeneration. The fitted artifact is byte-identical to the original; this was not a newly introduced fit change. Do not extrapolate the fitted model beyond the studied batch range.

## Reproduce the setup on a fresh checkout

Run these commands from the project root. The clone/environment creation steps are for a fresh setup, not for overwriting existing local folders. A compatible Windows Python 3.11 installation and Git are required.

```powershell
py -3.11 -m venv .venv
git clone --no-checkout https://github.com/gpu2grid/openg2g.git external/openg2g
git -C external/openg2g checkout --detach c582a5115cfd49dcdfb140f1837224e7cbf0cfb1
.\.venv\Scripts\python.exe -m pip install -r requirements-simulation.txt
.\.venv\Scripts\python.exe -m pip check
```

The [dataset](https://huggingface.co/datasets/ml-energy/benchmark-v3) requires approved access. If the machine is not already logged in, use the Hugging Face login prompt; do not paste a token into a tracked file:

```powershell
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe experiments\restore_simulation_data.py fetch
.\.venv\Scripts\python.exe experiments\restore_simulation_data.py prepare --verification results\simulator-restoration-repeat\verification.json
```

`fetch` downloads exactly eight named files and can reuse its cache. `prepare` uses the recovered original selection and fitting recipe and refuses to overwrite a populated prepared-data directory or verification file. In an already restored checkout, **skip preparation**; do not delete working data merely to repeat this step.

The current Windows setup used ordinary cached files rather than symlinks. The Hub's warning about extra cache space did not prevent the download; no machine-wide security or Developer Mode setting was changed.

## Run the comparison and tests

With the restored inputs already present:

```powershell
# A new output directory protects authoritative evidence from overwriting.
.\.venv\Scripts\python.exe experiments\slow_update_check.py --output results\slow-update-check-repeat
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests -q
```

The wrapper freezes its protocol, verifies source/numerical inputs, checks two exact historical trajectories, and only then runs four slower-controller cases. It stops if compatibility fails. The legacy `delay_screen.py` full-suite commands retain their original full-manifest hash check and will reject the newly timestamped manifest; use this documented wrapper for the restored comparison rather than disabling the old check.

Validation completed here: **32 tests and 25 subtests passed, with no skipped tests**. This is the project's test suite, not a rerun of every upstream OpenG2G test. Disabling pytest's optional cache avoids a pre-existing cache-directory permission warning; it does not disable tests.

Each 300-second simulated run took **5.8–8.9 seconds** and approximately **139–141 MiB** of sampled process-tree memory. The six child runs took approximately 40.4 seconds total. These measurements exclude dependency installation, downloading, input preparation, and report writing. The simulation replays measured H100 profiles; it does not measure the laptop's LLM throughput.

## References

- [Pinned OpenG2G source](https://github.com/gpu2grid/openg2g/tree/c582a5115cfd49dcdfb140f1837224e7cbf0cfb1), especially `InferenceData.generate`, `LogisticModelStore.generate`, and the README. The restored local code was inspected directly.
- [ML.ENERGY Benchmark v3](https://huggingface.co/datasets/ml-energy/benchmark-v3/tree/fea6cb99d037d11f86f9c45a6991b9e5764f4529): the pinned measured-data source.
- [Historical manifest](../results/power-model-pilot-verified/manifest.json): original revisions, versions, and prepared-data hashes.
- [Restoration script](../experiments/restore_simulation_data.py), [dependency pins](../requirements-simulation.txt), [restoration verification](../results/simulator-restoration/verification.json), and [restoration tests](../tests/test_restore_simulation_data.py).
- [Comparison protocol](../results/slow-update-check/protocol.json), [compatibility checks](../results/slow-update-check/compatibility.json), and [completed results](../results/slow-update-check/summary.json).
