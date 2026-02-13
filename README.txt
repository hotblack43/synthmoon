These are the portable “project capsule” text files for SYNTHMOON (currently v0.1.x).

They are meant to be handed to a new ChatGPT/AI session to restore context quickly.

Files:
- SPEC.md              Project specification + current v0.1.x design/behaviour
- scene.toml           Central configuration template (v0.1.2-capable)
- context_capsule.txt  Short pasteable summary for a fresh chat
- README.txt           This file
- write_project_files.py  Recreates the above files locally (no copy/paste)

Usage (to recreate locally):
    python write_project_files.py

Notes:
- The actual code lives in your git repo. These docs are the “why/what/how” memory.
- Keep large data (LROC/LOLA) out of git; use download scripts + .gitignore.
