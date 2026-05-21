# Project conventions

## Git commits — strict rules

All commits must be authored as **Devon Smith <devontjsmith@gmail.com>**.

Never add any of the following to commit messages:
- `Co-Authored-By: Claude` (or any AI co-author trailer)
- `Generated with Claude Code` (or any "generated with" footer)
- 🤖 or any robot emoji

The repo-local git config already sets author/committer identity — do not override
it with `--author` or `-c user.name=...` arguments. Commit messages are present-tense,
describe what changed and why, with no AI attribution.

## Code style

- Python 3.11; `pathlib.Path`, not `os.path.join`
- Type hints on function signatures; docstrings for non-trivial functions
- Format with `black` (line length 100); lint with `ruff` before committing
- `pytest tests/` must stay green

## Branch convention

- Commit to `main` directly (small private repo, three contributors)
- Do not force-push to `main` unless explicitly asked

## Layout

- `scripts/` runnable entry points · `src/` importable modules · `tests/` unit tests
- `configs/v0_default.yaml` holds v0 hyperparameters
- Large data lives under `/mnt/disks/splice-data`, never in git
