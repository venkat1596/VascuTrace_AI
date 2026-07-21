# Codex collaboration record

> Research prototype. Trained and evaluated using simulated vascular-like abnormalities, not confirmed human post-angioplasty lesions.

This is a curated public decision and evidence log. It records project roles,
decisions, observable outcomes, and links to reviewable artifacts. It does not
contain conversation turns, internal analysis, development-agent instructions
or configuration, local workstation details, credentials, or project-session
content.

Codex with GPT-5.6 served as the main planner, report writer, Git handler, and
primary technical decision maker within the development workflow. The project
record substantiates this role: Codex authored the initial project
specification, most Markdown planning documents in the local planning archive,
the public publication plan, and the technical report under `docs/report/`.
The owner retained final authority throughout the project.

The sanitized release publishes the planning and report artifacts that are
safe for public review. Development-only plans containing agent instructions
remain local and are not reproduced in the release.

## Chronological role and decision log

### 1. Research scope and claim boundary

Codex organized the initial product goal into a bounded method-development
question about image-domain synthetic-source detectability and deterministic
quantification in healthy PET/CT backgrounds. It carried the permanent warning
and nonclinical vocabulary into plans, reviews, the application, and the
technical report.

The owner decided the research scope and approved the claim boundary. The
result is visible in the [project overview](../README.md) and the
[technical report](report/VascuTrace_Technical_Report_2026-07-20.pdf).

### 2. Architecture and execution planning

Codex converted the approved scope into staged plans for physical-coordinate
PET/CT geometry, bilateral crops, controlled synthetic-source generation, a
transparent threshold path, an exploratory 2.5D Siamese model, deterministic
quantification, product orchestration, evaluation, and publication. Plans
separated intended interfaces from completed evidence and attached checks to
scientifically consequential work.

The owner approved or rejected those plans and retained the final product,
engineering, and design choices. The current implementation and remaining
gates are summarized in the [README](../README.md) and the
[publication and reproducibility plan](../plans/VascuTrace_Publication_and_Reproducibility_Plan_2026-07-20.md).

### 3. Bounded implementation

Claude was used only to write code from Codex-authored plans. Codex reviewed
the resulting implementation against the planned interfaces and acceptance
criteria, while the owner retained authority to accept, revise, or stop the
work. Claude did not choose the scientific claims, publication status,
licensing, repository visibility, competition category, or submission content.

Public examples of the implemented product surface are the
[dashboard entry point](../app.py), the
[product evaluation command](../scripts/run_product_evaluation.py), and the
[complete synthetic-case command](../scripts/run_complete_case.py).

### 4. Critical review and verification

Codex used explicit acceptance checks and independent review to compare the
implementation with the research contract. This review kept exploratory
center-slice observations separate from held-out, scan-level, native-space 3D,
or clinical results. It also preserved deterministic ownership of numeric and
laterality fields.

The review identified work that remains open: replacement of legacy reflection
geometry, completion of the promotion-compliant threshold baseline,
native-space 3D stitching, sealed subject-level evaluation, product integration
of the complete 3D quantifier, and rebuilding the public retrieval corpus. These
limits remain visible in the README and technical report rather than being
treated as completed results.

### 5. Technical report production

Codex organized the technical report around primary-source facts, historical
aggregate measurements, current implementation observations, design decisions,
and planned evaluations. It drafted and reviewed the report while preserving
the synthetic status, evidence provenance, limitations, and the owner's final
publication decision.

The frozen report artifacts are:

* PDF SHA-256:
  `f4fc1e790fa12c6da4139ad938fa12395955e1458d89ff7f44de6635b0033069`
* TeX SHA-256:
  `fc8837addd7ac829c256127003f50ccbd6c6e91bd5fed2a9d6130b5d3c66ea9b`

### 6. Git stewardship and release preparation

Codex served as the Git steward for scoped working-tree review, public-path
allowlisting, report-integrity checks, and sanitized release preparation. The
tracking policy excludes datasets, medical volumes, model weights, credentials,
generated outputs, private development material, and development records.

The sanitized `dev` release was published only after the owner authorized the
commit and push. The owner retains all future commit, push, licensing,
visibility, and sharing decisions.

### 7. Hackathon documentation

Codex converted the competition requirements into a public collaboration
summary, a reusable feature description, a timed audio demonstration outline,
an offline deterministic judge path, and an owner-completed submission
checklist. The repository URL is recorded. Category, video URL, license,
visibility, sharing status, and Codex Session ID fields remain explicitly
unresolved in the [submission guide](HACKATHON_SUBMISSION.md).

## Where Codex accelerated the workflow

Codex accelerated work through concrete coordination mechanisms:

* converting broad project goals into traceable requirements and acceptance
  checks;
* keeping implementation status, measured evidence, design decisions, and
  planned work distinct;
* coordinating implementation review and independent verification while Claude
  remained limited to planned code writing;
* assembling one evidence-controlled report from public sources, aggregate
  project evidence, and reviewed implementation behavior;
* maintaining repeatable commands, integrity hashes, and release checks; and
* translating submission rules into a reusable demonstration and judge-testing
  checklist.

No quantified time-saving claim is made.

## Human decisions and approvals

The project owner remained responsible for:

* the problem definition, product scope, and scientific boundaries;
* selection and identification of Codex with GPT-5.6;
* approval or rejection of plans, implementation candidates, checks, and report
  language;
* the final product, engineering, design, and scientific decisions;
* report publication and the decision to release any repository state;
* licensing, repository visibility, judge access, and sharing;
* competition category, video production, and public video URL; and
* the final submission and Codex Session ID provided to the organizer.

Codex supplied structured decision support. It did not replace these human
approvals.

## Public product code and private development material

VascuTrace product GenAI prompts are shipped application code and remain part
of the product source selected for a sanitized release. They are not reproduced
in this document. Private development-agent artifacts, including definitions,
instructions, configurations, and development records, are excluded from the
public release and are not reproduced here.
