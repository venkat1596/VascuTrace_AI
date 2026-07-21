# VascuTrace AI

> Research prototype. Trained and evaluated using simulated vascular-like abnormalities, not confirmed human post-angioplasty lesions.

VascuTrace is a reproducible PET/CT method-development prototype. It studies
whether controlled vascular-like synthetic sources can be detected and
quantified in healthy PET/CT backgrounds. It is not a diagnostic system and no
reported result establishes clinical sensitivity, clinical specificity, or
patient benefit.

The detailed technical report is available at
[docs/report/VascuTrace_Technical_Report_2026-07-20.pdf](docs/report/VascuTrace_Technical_Report_2026-07-20.pdf).
It includes aggregate EDA, method schematics, actual generated product views,
the verified five-tool runtime trace, all six product checks, corrected
development-collaboration evidence, and a sanitized analysis of 11 root Codex
sessions. No patient image or model weight is used in the report evidence
build.

## Current implementation

The repository contains:

* PET/CT geometry utilities that use physical patient coordinates and named
  transforms
* subject-grouped data contracts and deterministic bilateral crop generation
* a parameterized image-domain synthetic-source engine
* a transparent threshold baseline
* deterministic 3D quantification with structured null and QC results
* a 2.5D shared-weight Siamese U-Net training and evaluation path
* a research-demonstrator application with deterministic tools, MCP exposure,
  optional local evidence retrieval, report generation, and numeric-fidelity
  verification

The product workflow keeps measurement code separate from generated prose.
Language generation cannot create or replace quantitative values. The default
report backend is a deterministic template, and the default detection backend
is a synthetic-reference path intended for integration testing. The trained
Siamese backend is opt-in and currently processes a selected cached 2D
validation sample rather than a complete scan.

The current exploratory B2 result was measured on 208 validation center slices,
including 78 positive and 130 negative slices, drawn from seven subject
clusters. At the frozen operating point, positive-slice mean IoU was 0.614895,
75 of 78 positive slices had a target-overlapping prediction, and 37 of 130
negative slices contained activation. These are validation-only 2D observations,
not held-out test, scan-level, 3D, or clinical performance estimates.

## Setup

The project targets Python 3.13 and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --locked
uv run ruff check --no-cache .
uv run ruff format --check --no-cache .
uv run pytest -q -m "not local_data and not gpu" \
  -k "not test_dataloader_with_multiple_workers"
```

CPU and offline tests use generated fixtures. Dataset files, medical volumes,
model weights, caches, credentials, and run outputs are not versioned.

The multiprocessing DataLoader node is verified separately because restricted
containers may not allow worker processes to complete. During release review,
it reached a 90-second cap without pytest failure output. On a host that permits
multiprocessing, run:

```bash
uv run pytest -q \
  tests/test_ml_dataset.py::TestPicklingAndDataLoader::test_dataloader_with_multiple_workers
```

## Research demonstrator

Run the deterministic local dashboard:

```bash
uv run streamlit run app.py
```

Run the product evaluation and complete synthetic case paths:

```bash
uv run python -m scripts.run_product_evaluation
uv run python -m scripts.run_complete_case
```

Rebuild the generated-only product receipt and report figures 09 through 11:

```bash
uv run python docs/report/scripts/build_product_evidence.py
```

Run the MCP server over standard input and output:

```bash
uv run python -m vascutrace.mcp_server
```

Generated artifacts are written under the configured output root and remain
untracked.

## Optional product backends

Every optional backend is explicitly selected. Offline deterministic behavior
is the default.

| Setting | Default | Optional value |
|:--|:--|:--|
| `VASCUTRACE_DETECTION_BACKEND` | `reference` | `siamese` |
| `VASCUTRACE_REPORT_BACKEND` | `template` | `llm` |
| `VASCUTRACE_EVIDENCE_BACKEND` | `keyword` | `rag` |

The optional report path uses an OpenAI reasoning model for interpretation and
local Qwen models for embedding and reranking. Deterministic code owns all
measurements and laterality fields. The public retrieval corpus must be rebuilt
locally before enabling RAG because generated indices are not versioned.

## Collaboration with Codex

Codex with GPT-5.6 performed every non-coding workflow role in the VascuTrace
development process. This included planning and architecture, primary technical
decisions, scientific review, report writing, delivery review, Git and release
handling, and humanizer and editorial review. Codex also authored the plans and
instructions used for code implementation. Claude was used only to implement
that planned code. The project owner retained final authority over all product,
scientific, publication, licensing, repository, category, video, and submission
decisions.

The public evidence projection covers 11 root Codex sessions through
`2026-07-21T14:56:53.784Z`. Their metadata records `gpt-5.6-sol` in all 11
sessions. The projection contains 87 user turns, 725 assistant updates, 69
started tasks, 60 completed tasks, 5,654 tool-call events, 654 patch events,
629 bounded review activities, 77 web searches, and 38 context compactions.
These are structural event counts, not measures of quality, labor time, or
scientific performance.

VascuTrace product GenAI prompts are shipped application code. Private
development-agent artifacts are excluded from the public release. The
[session-based collaboration record](docs/CODEX_COLLABORATION.md) documents the
public decision and evidence trail, links the sanitized
[session receipt](docs/report/evidence/codex_session_evidence.json), and shows
the collaboration, timeline, and activity figures. The
[hackathon submission guide](docs/HACKATHON_SUBMISSION.md) collects the
description, demonstration, testing, and owner-completed submission fields.
The selected competition category is `Work, Life and Productivity`.

## Scientific status and limitations

The implemented components are not yet a fully integrated scientific pipeline.
Important open work includes:

* replacing the legacy bilateral-reflection crop method with the frozen
  iliac-only physical-coordinate method
* completing a promotion-compliant threshold baseline
* stitching model outputs into native-space 3D masks
* running subject-clustered evaluation on a sealed test split
* integrating the standalone 3D quantifier into the product path
* rebuilding and reevaluating the sanitized public retrieval corpus

Invalid or unavailable scientific measurements should be represented as
structured nulls with explicit QC reasons, never silently converted to zeros.

## Branches

* `main` is the release branch.
* `dev` is the active development branch.

CI runs lockfile, lint, formatting, and test checks for pushes and pull requests.
