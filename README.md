# VascuTrace_AI

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 (pinned in `.python-version`).

```bash
uv sync          # creates .venv and installs dependencies
uv run pytest    # run the tests
```

## Branches

- `main` — protected release branch. Code lands here only via a pull request from `dev`
  with CI green.
- `dev` — the working branch. All development happens here.

## CI

`.github/workflows/ci.yml` runs on every push to `main`/`dev` and on every PR:
lockfile freshness (`uv sync --locked`), `ruff check`, `ruff format --check`, `pytest`.

## Not in this repo

Datasets, model weights/checkpoints, run outputs, `.env` secrets, and the local Claude
harness (`.claude/`, `claude-harness/`) are all gitignored — see `.gitignore`.
