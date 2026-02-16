# synthmoon — working agreements for Codex

- This repo is run with `uv run ...` (do not use bare `python` unless asked).
- Prefer minimal, surgical changes. Avoid wide refactors unless explicitly requested.
- Do not add new heavy dependencies without asking.
- Preserve existing functionality; add tests or a small reproducible check if you change behaviour.
- When changing rendering/layers, add a tiny sanity check that verifies no layer becomes identically zero unless intended.
- Use patch/diff style outputs in explanations when possible.
- Assume we only run via ./run_looper_hourly.sh
