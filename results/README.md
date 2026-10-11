# Robotics experiment evidence

## Laptop feasibility

- [robotics-feasibility/summary.json](robotics-feasibility/summary.json): authoritative working environment, pinned model identity, simulator assertions, GPU/RAM measurements, prediction timing and three complete baseline episodes.
- [robotics-feasibility/simulator-check.json](robotics-feasibility/simulator-check.json): earlier simulator-only smoke test. Its package inventory predates the PyArrow/pandas compatibility correction; use `summary.json` for the final environment.
- [Feasibility report](../reports/2026-10-10-robotics-laptop-feasibility.md): understandable conclusions, measurement definitions, limitations and reproduction instructions.

These results establish offline laptop feasibility, not a reliable success-rate estimate or an improvement from our proposed controller. Preserve the original evidence; put disposable reruns in a `*-repeat/` directory, which is Git-ignored.

Power-grid results belong to the repository's `main` branch and are intentionally absent here.
