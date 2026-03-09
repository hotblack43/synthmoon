# README_RESUME_WORK

Quick commands to resume work safely on the correct branch.

## 1) Resume on this machine (after reboot)

```bash
cd /home/pth/WORKSHOP/synthmoon
git fetch origin
git switch feature/earthlight-next
git status -sb
uv run python -m synthmoon.run_v0 --help
```

## 2) Resume on another machine

```bash
git clone ssh://git@github.com/hotblack43/synthmoon.git
cd synthmoon
git fetch origin
git switch feature/earthlight-next
git status -sb
uv run python -m synthmoon.run_v0 --help
```

If the branch is not yet local:

```bash
git switch -c feature/earthlight-next --track origin/feature/earthlight-next
```

## 3) First-time branch setup (do once)

If you have not created/published the branch yet:

```bash
cd /home/pth/WORKSHOP/synthmoon
git switch -c feature/earthlight-next
git push -u origin feature/earthlight-next
```

## 4) Milestone protection (recommended)

Tag the current milestone before new development:

```bash
git tag -a v0.4.0-milestone -m "Milestone: advanced DEM+Hapke + single-image CLI"
git push origin v0.4.0-milestone
```
