SYNTHMOON patch: --layer5-only (write only layer 5)

What this does
- Default behaviour unchanged: writes ALL layers to the FITS cube.
- New CLI switch --layer5-only: filters the cube to keep only layer 5 (1-based) right before writing.

Files in this zip
- 0001_add_layer5_only_flag.patch   (adds CLI flag + filtering helper)
- 0002_make_layer_names_consistent.patch (optional: makes LAY1.. keywords robust after filtering)

How to apply (git repo)
1) In the root of your SYNTHMOON repo:

   unzip synthmoon_layer5_only_patch.zip
   git apply 0001_add_layer5_only_flag.patch
   git apply 0002_make_layer_names_consistent.patch   # optional but recommended

2) Run (example):
   PYTHONPATH=$PWD uv run python -m synthmoon.run_v0 --config scene.toml --out out_alllayers.fits
   PYTHONPATH=$PWD uv run python -m synthmoon.run_v0 --config scene.toml --out out_layer5.fits --layer5-only

If 'git apply' fails
- Your filenames likely differ (e.g., run module is synthmoon/run_v0.py vs synthmoon/run.py).
- In that case: open the patch and manually move the edits into your real file(s),
  or tell me your actual module path and I will regenerate a zip that matches exactly.
