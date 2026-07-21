# Hackathon submission guide

> Research prototype. Trained and evaluated using simulated vascular-like abnormalities, not confirmed human post-angioplasty lesions.

This guide is ready to adapt for the submission form and demonstration video.
Fields that require an owner decision or an external action remain unfilled.

## Submission fields

* Competition category: `[Owner entry required]`
* Repository URL: `https://github.com/venkat1596/VascuTrace_AI`
* Public YouTube URL: `[Owner entry required]`
* Codex Session ID or organizer-requested feedback identifier:
  `[Owner entry required]`
* External weights or large non-sensitive assets on Google Drive:
  `[Owner entry required]`
* Repository visibility and judge-sharing status:
  `[Owner verification required]`
* License: `[Owner selection required]`

No category, URL, identifier, visibility state, sharing state, or license is
inferred by this repository.

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

Codex with GPT-5.6 served as the main planner, report writer, Git handler, and
primary technical decision maker within the development workflow. It converted
the research goals into bounded plans, coordinated implementation and
independent checks, reviewed scientific claims and implementation gaps,
assembled the technical report, and prepared release and submission materials.
This role is documented by the Codex-authored initial specification and planning
archive, the public publication plan, the technical report under `docs/report/`,
and the curated chronological record.
Claude was used only to write code from Codex-authored plans. The owner retained
the final product, engineering, scientific, publication, licensing, repository
visibility, category, video, and submission decisions.

See the [curated collaboration record](CODEX_COLLABORATION.md) for the public
chronology and evidence links. VascuTrace product GenAI prompts are shipped
application code. Private development-agent artifacts are excluded, and
neither class of instructions is reproduced in this guide.

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
5. `1:55 to 2:25` Explain the collaboration. State that Codex with GPT-5.6
   served as the main planner, report writer, Git handler, and primary technical
   decision maker within the development workflow. State that Claude only
   implemented code from Codex-authored plans and that the owner retained final
   decisions.
6. `2:25 to 2:50` Close with reproducibility and limitations. Show the technical
   report, the frozen report hashes, the exploratory learned-model status, and
   the deterministic judge path.

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

1. Open the project thread where the majority of the core functionality was
   built.
2. Run `/status` and record the displayed chat ID separately.
3. Run `/feedback` to open the feedback dialog.
4. Review the information the dialog proposes to include before submitting it.
5. Complete the feedback flow and record the identifier or receipt it provides,
   if any.
6. Check the submission form wording and enter the exact identifier it requests
   in the Session ID field.

The official reference does not establish that `/feedback` itself returns a
Session ID. Do not silently substitute the chat ID, a feedback receipt, or
another identifier without confirming what the organizer requests. The
repository does not read Codex account or thread state and cannot fill this
field.

## Supported platform status

The declared software target is Python 3.13 or newer with `uv` for dependency
management. The judge path is designed for CPU and offline execution with
generated fixtures after dependencies are installed. It does not require raw
medical data, model weights, CUDA, an API key, a hosted test account, or model
training.

An operating-system compatibility matrix has not been independently verified,
and no hosted demo, sandbox, or test account is claimed. If the owner selects a
Plugins or Dev Tools category, confirm that the local deterministic path meets
the category requirement or provide an organizer-acceptable hosted option.

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

For this release, that command passed 732 tests, with 17 skipped and 11
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

* Select the category that best fits the project.
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
* Complete the `/feedback` workflow and enter the organizer-requested Session
  ID or feedback identifier.
* Confirm that the README links this guide and the collaboration record.
