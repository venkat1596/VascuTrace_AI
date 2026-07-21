# Hackathon submission guide

> Research prototype. Trained and evaluated using simulated vascular-like abnormalities, not confirmed human post-angioplasty lesions.

This guide is ready to adapt for the submission form and demonstration video.
Fields that require an owner decision or an external action remain unfilled.

## Submission fields

* Competition category: `Work, Life and Productivity`
* Repository URL: `https://github.com/venkat1596/VascuTrace_AI`
* Public YouTube URL: `Pending owner publication`
* Primary Codex Session ID:
  `019f6187-81f0-7db1-b5c1-d9cdbba313dc`
* Primary-thread `/feedback` receipt or organizer-requested identifier:
  `Pending owner feedback submission`
* Technical report authors: Venkat Sumanth Reddy Bommireddy; Prashanth Reddy
  Biyyani; Sukeshwar Bogundula; SAI Deeraj Bogundula
* External weights or large non-sensitive assets on Google Drive:
  `Optional; pending owner authorization and packaging`
* Repository visibility and judge-sharing status:
  `Pending owner verification`
* License: `Pending owner selection`

The competition category and root Codex chat/session IDs are owner-approved.
The video URL, feedback receipt, external-asset link, visibility and sharing
state, and license still require owner action.

## Codex session evidence for judging

Evidence class `SESSION-001` is a sanitized projection of the 11 allowlisted
root Codex sessions used for VascuTrace. The inclusive evidence cutoff is
`2026-07-21T14:56:53.784Z`. Every session records provider `openai` and model
identifier `gpt-5.6-sol`. Nine sessions used the Codex command-line interface;
the primary session and one supporting session used the Codex VS Code surface.

The primary Session ID is
`019f6187-81f0-7db1-b5c1-d9cdbba313dc`. It spans the main transition from
specification and EDA reconciliation through Phase 1 geometry delivery and
model-readiness review. It has the largest tool-call count, assistant-update
count, and bounded-review count in this allowlisted record: 1,224, 147, and 160,
respectively. All eight of its recorded tasks completed. Other sessions have
the largest patch count or task count, so this selection is based on its role in
the foundation-to-ML interval and its combined activity, not on one universal
ranking.

### Root session inventory

| Ref | Root Session ID | UTC evidence window | Surface | Model | Observed span |
|:--|:--|:--|:--|:--|--:|
| S01 | `019f5c4e-1f8d-7190-8a04-2c6f7e7a2e18` | 2026-07-13 16:37:07 to 17:43:57 | CLI | `gpt-5.6-sol` | 1.11 h |
| S02 | `019f5c9a-0be0-7351-b9c1-9d7b342d8108` | 2026-07-13 17:50:27 to 2026-07-14 10:02:46 | CLI | `gpt-5.6-sol` | 16.21 h |
| S03 | `019f6015-b646-70b3-ab11-6d5b20d9eeee` | 2026-07-14 10:10:08 to 12:40:21 | CLI | `gpt-5.6-sol` | 2.50 h |
| S04 | `019f60c0-ba06-77f0-be21-37da29651799` | 2026-07-14 13:11:12 to 16:29:04 | CLI | `gpt-5.6-sol` | 3.30 h |
| S05, primary | `019f6187-81f0-7db1-b5c1-d9cdbba313dc` | 2026-07-14 16:48:42 to 2026-07-15 05:29:37 | VS Code | `gpt-5.6-sol` | 12.68 h |
| S06 | `019f661f-b909-7a22-b165-7d5c8e5da35f` | 2026-07-15 14:13:42 to 20:07:47 | CLI | `gpt-5.6-sol` | 5.90 h |
| S07 | `019f6773-1bc8-73d2-9a39-b7ade344260f` | 2026-07-15 20:23:48 to 2026-07-16 01:22:55 | CLI | `gpt-5.6-sol` | 4.99 h |
| S08 | `019f6bd0-46c4-7121-bf6d-88481daf4566` | 2026-07-16 16:44:00 to 16:56:43 | VS Code | `gpt-5.6-sol` | 0.21 h |
| S09 | `019f6d22-f305-7ca0-86ae-20e1ac55dba6` | 2026-07-16 22:57:24 to 2026-07-17 23:28:33 | CLI | `gpt-5.6-sol` | 24.52 h |
| S10 | `019f7398-2350-79d0-a657-fca01373af04` | 2026-07-18 05:00:24 to 2026-07-19 17:47:26 | CLI | `gpt-5.6-sol` | 36.78 h |
| S11 | `019f81d0-b3be-7e21-a19a-491a13e360e8` | 2026-07-20 23:23:20 to 2026-07-21 14:56:53 | CLI | `gpt-5.6-sol` | 15.56 h |

Observed span is the difference between the first and last allowlisted record
in a session. The sum, 123.76 hours, is not labor time because sessions can be
idle or overlap.

### Session activity counts

| Ref | User turns | Assistant updates | Tasks started | Tasks completed | Tool calls | Patch events | Bounded review activities | Web searches | Compactions |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| S01 | 2 | 12 | 1 | 1 | 50 | 0 | 11 | 17 | 1 |
| S02 | 8 | 72 | 4 | 2 | 994 | 309 | 83 | 25 | 7 |
| S03 | 6 | 31 | 4 | 4 | 156 | 8 | 21 | 2 | 1 |
| S04 | 6 | 45 | 4 | 2 | 204 | 10 | 24 | 3 | 1 |
| S05, primary | 10 | 147 | 8 | 8 | 1,224 | 92 | 160 | 0 | 6 |
| S06 | 9 | 59 | 7 | 6 | 461 | 29 | 60 | 0 | 2 |
| S07 | 4 | 23 | 4 | 4 | 151 | 13 | 9 | 0 | 2 |
| S08 | 1 | 6 | 1 | 1 | 46 | 0 | 3 | 5 | 0 |
| S09 | 21 | 91 | 18 | 17 | 501 | 17 | 91 | 17 | 7 |
| S10 | 10 | 113 | 10 | 9 | 827 | 91 | 54 | 0 | 5 |
| S11 | 10 | 126 | 8 | 6 | 1,040 | 85 | 113 | 8 | 6 |
| **Total** | **87** | **725** | **69** | **60** | **5,654** | **654** | **629** | **77** | **38** |

The projector counts user-turn and assistant-update events without copying
their message text. Task counts come from task lifecycle events. Tool calls,
patches, bounded review activities, completed web searches, and context
compactions are structural event categories. The receipt also records 28,258
source records at the cutoff, 107 model-context records, and six MCP tool calls.
These counts document workflow activity. They do not measure quality,
productivity, labor time, authorship effort, or scientific performance.

### Session task themes and public outcomes

| Ref | Main task theme copied from the receipt | Representative public outcome copied from the receipt |
|:--|:--|:--|
| S01 | Translate the initial specification into a two-track PET/CT architecture with fixed interfaces. | Architecture and implementation inventory: `README.md`; `plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md` |
| S02 | Develop a Codex-led seven-day plan with scientific, product, evaluation, and submission gates. | Publication and verification gates: `plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md` |
| S03 | Perform local EDA from imaging, ML, mathematical, and statistical perspectives. | Reproducible EDA entry point: `scripts/eda_quadra.py` |
| S04 | Define an acceptance-first Phase 1 foundation for provenance, geometry, QA, splitting, and resources. | Affine-aware PET-grid geometry: `src/vascutrace/geometry.py`; `tests/test_geometry.py` |
| S05, primary | Reconcile the specification, EDA, and product workplan after resuming from Git. | Physical-grid geometry: `src/vascutrace/geometry.py`; `tests/test_geometry.py` |
| S06 | Report the concrete status and contents of Phase 1. | Executable Phase 2 data pipeline: `scripts/run_p2_pipeline.py`; `src/vascutrace/data/ingest.py`; `src/vascutrace/data/crops.py` |
| S07 | Resolve remaining Phase 1 binding and completion blockers. | Controlled source simulation: `src/vascutrace/simulation/anomaly.py`; `tests/test_simulation.py` |
| S08 | Assess whether the reported IoU was near a defensible maximum. | Failure-specific evaluation: `src/vascutrace/ml/evaluate.py`; `src/vascutrace/ml/metrics.py`; `tests/test_ml_evaluate.py` |
| S09 | Falsify unsupported IoU-ceiling claims and correct statistical and reproducibility defects. | Boundary and soft-target losses: `src/vascutrace/ml/losses.py`; `tests/test_ml_boundary_aux.py`; `tests/test_ml_b3_soft_term.py` |
| S10 | Integrate reviewed planning, source-restoration, and validation repairs. | Config-gated training and evaluation: `src/vascutrace/ml/losses.py`; `src/vascutrace/ml/train.py`; `src/vascutrace/ml/evaluate.py` |
| S11 | Produce a professional EDA-to-pipeline report with figures and independent checks. | Technical report: `docs/report/VascuTrace_Technical_Report_2026-07-20.tex`; `docs/report/VascuTrace_Technical_Report_2026-07-20.pdf` |

The full sanitized themes and public outcomes are in
[`codex_session_evidence.json`](report/evidence/codex_session_evidence.json).
The projection reads only allowlisted metadata and structural event types. It
does not publish message bodies, transcripts, hidden reasoning, system or
developer text, prompts, private instructions, tool arguments or results,
credentials, workstation state, or absolute source paths. Raw JSONL session
files remain local and are not part of the repository.

## Ready-to-adapt feature description

VascuTrace is a reproducible PET/CT method-development prototype for controlled
image-domain experiments. It studies whether simulated vascular-like sources
can be detected and deterministically quantified in healthy PET/CT backgrounds.
The repository includes physical-coordinate geometry utilities, a parameterized
synthetic-source engine, a transparent threshold path, an exploratory 2.5D
shared-weight Siamese U-Net path, standalone deterministic 3D quantification
with structured QC results, a Streamlit research demonstrator, evidence
retrieval, bounded report generation, numeric-fidelity verification, and an
executable product evaluation.

The default demonstration uses generated fixtures and deterministic reference
backends. It separates code-owned measurements and laterality from optional
generated prose. The learned backend remains exploratory and currently handles
a selected cached 2D validation sample rather than a complete scan. VascuTrace
does not establish diagnosis, clinical sensitivity or specificity,
arterial-wall ground truth, scanner sensitivity, treatment response, or patient
outcomes.

## Codex collaboration statement

Codex with GPT-5.6 performed every non-coding workflow role in the development
process. It owned planning and architecture, primary technical decisions,
scientific review, report writing, delivery review, Git and release handling,
and humanizer and editorial review. Codex also authored the plans and
instructions used for code implementation. Claude was used only to implement
that code. Scientific review, delivery review, and humanizer review were
therefore Codex with GPT-5.6
workstreams, not Claude coding responsibilities or VascuTrace runtime agents.
The owner retained final product, engineering, scientific, publication,
licensing, repository visibility, category, video, and submission authority.

This attribution is supported by the recorded `gpt-5.6-sol` model identifier
across all 11 root sessions, the session-derived task and activity record above,
the Codex-authored initial specification and planning archive, the public
publication plan, the technical report, and the chronological collaboration
record.

See the [session-based collaboration record](CODEX_COLLABORATION.md) for the
public chronology, collaboration figure, timeline, activity analysis, and
evidence links. VascuTrace product GenAI prompts are shipped application code.
Private development-agent artifacts are excluded, and neither class of
instructions is reproduced in this guide.

## Audio demonstration outline

Target duration: 2 minutes 50 seconds. Rehearse the final recording and keep it
below 3 minutes.

1. `0:00 to 0:20` Introduce the bounded research question and read the permanent
   warning. Show the README title and warning while the audio states that the
   project studies simulated vascular-like abnormalities, not clinical
   diagnosis.
2. `0:20 to 0:45` Show the repository overview. Explain physical-coordinate
   PET/CT utilities, controlled source simulation, deterministic measurement,
   the research demonstrator, and executable evaluation.
3. `0:45 to 1:30` Run or open the deterministic Streamlit dashboard. With clear
   audio, show the generated case overview, PET/CT views, simulated mask,
   structured measurements, report verification, evidence area, and audit
   trace. Do not present the reference backend as a trained complete-scan
   result.
4. `1:30 to 1:55` Show the product evaluation and complete synthetic-case
   commands. Explain that judges can exercise the default path without raw
   medical data, model weights, CUDA, an API key, or model retraining.
5. `1:55 to 2:25` Explain the collaboration while showing Figures 11 through
   13. State that Codex with GPT-5.6 performed planning, primary technical
   decisions, scientific review, report writing, delivery review, Git and
   release handling, and humanizer review, and authored the implementation
   instructions. State that Claude only implemented code and that the owner
   retained final authority.
6. `2:25 to 2:50` Close with reproducibility and limitations. Show the technical
   report, the sanitized session evidence, the exploratory learned-model status,
   and the deterministic judge path.

The video must have clear audio. Upload it to YouTube, make it publicly visible,
and enter its public URL in the submission form.

## Copyright and trademark check

Use only narration, music, images, video, fonts, logos, and other material that
the owner created or has permission to use. Do not add copyrighted music or
third-party media without permission. Use third-party product names only as
needed for accurate attribution, and do not display third-party trademarks or
brand assets unless their terms or permission allow it. Check the desktop,
browser tabs, notifications, and recording background for unrelated protected
or private material before recording.

## Codex feedback and identifier workflow

The official [slash-command reference](https://learn.chatgpt.com/docs/reference/slash-commands)
states that `/feedback` opens the feedback dialog and may include logs. It also
states that `/status` shows the chat ID.

The IDs and counts in this guide come from the sanitized `SESSION-001`
projection. The projection uses allowlisted metadata, event categories, curated
task themes, and public artifact outcomes through the fixed cutoff. It does not
publish raw conversation text, private prompts, hidden reasoning, tool content,
or subagent IDs. The primary ID is the 2026-07-14 VS Code root thread selected
for its central foundation-to-ML role and its combined structural activity.

1. Open the primary project thread with Session ID
   `019f6187-81f0-7db1-b5c1-d9cdbba313dc`.
2. Run `/status` and confirm that the displayed chat ID matches the primary ID.
3. Run `/feedback` to open the feedback dialog for that primary thread.
4. Review the information the dialog proposes to include before submitting it.
5. Complete the feedback flow and record the identifier or receipt it provides,
   if any.
6. Check the submission form wording and enter the primary Session ID, feedback
   receipt, or both exactly as requested. Supporting IDs are supplemental and
   do not replace the primary-thread `/feedback` action.

The official reference does not establish that `/feedback` itself returns a
Session ID. Do not silently substitute the chat ID, a feedback receipt, or
another identifier without confirming what the organizer requests. The release
contains the sanitized receipt, figures, copied identifiers, and this workflow.
It does not contain raw session files, transcripts, prompts, or private Codex
logs.

## Supported platform status

The declared software target is Python 3.13 or newer with `uv` for dependency
management. The judge path is designed for CPU and offline execution with
generated fixtures after dependencies are installed. It does not require raw
medical data, model weights, CUDA, an API key, a hosted test account, or model
training.

An operating-system compatibility matrix has not been independently verified,
and no hosted demo, sandbox, or test account is claimed. The selected category
is `Work, Life and Productivity`; judges can use the local deterministic path
without rebuilding a model.

## Installation

Install Python 3.13 or newer and
[`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
```

The dependency download may require package-index access on a fresh machine.
The deterministic commands below make no network or API request once the
locked environment is installed.

## Deterministic judge test

Run the product evaluation:

```bash
uv run python -m scripts.run_product_evaluation \
  --output outputs/judge_evaluation/summary.json
```

A successful run exits with code 0, reports zero failed product checks, and
writes the structured summary at the requested output path.

Generate a complete synthetic case and artifact manifest:

```bash
uv run python -m scripts.run_complete_case \
  --output-dir outputs/judge_case
```

This path uses generated inputs and does not require a dataset download or
model weight. It exercises product orchestration, deterministic measurements,
report generation, verification, and artifact creation without retraining.

Run the CPU and offline regression suite:

```bash
uv run pytest -q -m "not local_data and not gpu" \
  -k "not test_dataloader_with_multiple_workers"
```

For this release, the equivalent tracked offline selection was run in bounded
batches: 758 tests passed, one was skipped, and collection reported 11
deselected by markers or the explicit multiprocessing exclusion. The separate
DataLoader worker node reached a 90-second cap in the constrained release
container without pytest failure output. On a host that permits
multiprocessing, run it separately:

```bash
uv run pytest -q \
  tests/test_ml_dataset.py::TestPicklingAndDataLoader::test_dataloader_with_multiple_workers
```

Optionally open the local demonstrator:

```bash
uv run streamlit run app.py
```

Use the default backends for judging. Optional learned, language-model, and RAG
backends have additional local assets or service requirements and are not part
of the deterministic judge path.

## External weights and large assets

Model weights and large data files are excluded from Git and from the published
`dev` history. The deterministic judge path above does not require them.

The owner may add a Google Drive link for weights or other large assets only
when those files are non-sensitive and authorized for redistribution. Record
the exact file names, versions, checksums, license or access terms, and download
instructions beside that link. Raw medical volumes, headers, identifiers, and
case-level records remain local and must not be uploaded to Google Drive or any
other external service.

## Repository access, license, and sharing actions

No license selection or repository visibility state is asserted here. Before
submission, the owner must:

1. Choose and add a license appropriate for the intended public distribution.
2. Enter the exact judge-facing repository URL.
3. Make the repository public with the relevant license, or keep it private and
   share access with `testing@devpost.com` and
   `build-week-event@openai.com` as required by the competition rules.
4. Verify from a clean judge-like environment that the repository can be
   accessed, installed, and tested using the documented deterministic path.
5. Record the final visibility and sharing status in the submission form.

These are owner-controlled external actions. This guide does not change the
repository, its visibility, collaborator access, or a submission form.

## Final submission check

* Confirm the selected `Work, Life and Productivity` category in the submission
  form.
* Adapt the feature description without weakening the scientific boundary.
* Record a clear audio demonstration shorter than 3 minutes.
* Confirm media permission and remove unrelated trademarks or protected
  material.
* Publish the video on YouTube and enter its public URL.
* Provide the accessible repository URL and complete the license or private
  judge-sharing action.
* Add the owner-controlled Google Drive link and checksums only for authorized,
  non-sensitive external assets; keep raw medical data local.
* Run and record the deterministic judge test result.
* Complete `/feedback` in the primary thread and enter the organizer-requested
  primary Session ID or feedback identifier. Keep supporting IDs supplemental.
* Confirm that the README links this guide and the collaboration record.
