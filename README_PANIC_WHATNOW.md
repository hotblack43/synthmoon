# README_PANIC_WHATNOW (SYNTHMOON)

If you feel lost: you are not lost. Use the script. It puts you back on rails.

## The one command
From the repo folder:

    cd ~/WORKSHOP/SYNTHMOON
    ./panic_run_v0_1.sh

What it does:
- stashes any local edits (so nothing breaks)
- checks out the known-good tag `v0.1.0`
- runs `uv sync`
- renders using `scene.toml`
- saves everything into a timestamped folder under `OUTPUT/runs/...`
  - config copy
  - git version
  - log
  - output FITS

## Where is my output?
Each run creates a folder like:

    OUTPUT/runs/20260213T081500Z/

Inside:
- scene.toml (exact config used)
- git_version.txt (exact code revision)
- run.log (terminal output)
- output FITS copied in

## Different config / different tag
    ./panic_run_v0_1.sh path/to/scene.toml
    ./panic_run_v0_1.sh scene.toml v0.1.0

## I edited things and regret it
The script stashes automatically. To see stashes:

    git stash list

To restore the most recent stash:

    git stash pop

## If SPICE kernels break again
Usually:
    uv run python scripts/download_kernels.py

Then rerun:
    ./panic_run_v0_1.sh
