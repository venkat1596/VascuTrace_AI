# VascuTrace publication and reproducibility plan

Date: 2026-07-20

> Research prototype. Trained and evaluated using simulated vascular-like abnormalities, not confirmed human post-angioplasty lesions.

## Purpose

This plan defines the public path from the current research prototype to a reproducible methods package. It covers the technical report, aggregate evidence, code verification, scientific acceptance gates, and a sanitized development-branch release. It contains no private development configuration, prompt, runtime log, or model weight.

## Publication package

The release package should contain:

1. The installable VascuTrace source and product orchestration code.
2. Generated-fixture tests, CPU and offline regression tests, and clearly marked optional data or GPU tests.
3. Public configuration files required to reproduce model architecture and intended training settings.
4. The technical report source, bibliography, machine-readable aggregate evidence ledger, generated figure script, eight publication figures, and compiled PDF.
5. Public algorithm notes and this project-facing plan.
6. A README that distinguishes the deterministic reference fixture from the optional learned backend.

The release must not contain medical volumes, identifiers, per-case tables, model weights, run outputs, credentials, caches, local retrieval indices, or workstation-specific material.

## Frozen scientific language

Public artifacts use `abnormality_score` for an uncalibrated model output. They describe target-overlapping predictions on synthetic inputs and activation on negative healthy-control backgrounds. They do not report clinical diagnosis, clinical sensitivity or specificity, arterial-wall truth, scanner sensitivity, patient-motion or attenuation-correction simulation, treatment response, or outcomes.

The exact research warning appears in the report, application, structured reports, and relevant documentation.

## Evidence model

Each report result belongs to one class:

- Primary-source fact, with a direct citation.
- Historical aggregate measurement, with an input hash and an explicit note that it was not rerun during publication.
- Current implementation observation, tied to reviewed code.
- Design decision, stated as a rule rather than a result.
- Planned evaluation, stated in future tense.

The aggregate evidence ledger records the population, independent unit, value, units, status, source hash, and limitation for every public number used in a figure.

## Gate 1: dataset provenance and data fitness

Acceptance requires:

- Zenodo record 16686025 version 1.1.1 and MD5 `5b37cd936988a023c67fe8f85d634d41` recorded as the current archive source.
- The archive-era MOOSE label map pinned independently before final corridor promotion.
- Strict subject, session, modality, and derivative inventory.
- Finite, invertible affines and explicit physical-coordinate transforms.
- Local fused PET, CT, and iliac-anchor QA on the promoted corridor.
- Subject-grouped, sex-stratified split with both visits and all derivatives together.

Raw data remain local and are never part of the public package.

## Gate 2: geometry and crop contract

Replace the legacy reflection method with the intended iliac-only method:

1. Extract and freeze bilateral iliac centerlines.
2. Pair normalized arc-length positions over their shared usable extent.
3. Orient every left-to-right pair toward patient RAS right.
4. Fit a resistant mean normal and median plane offset.
5. Validate side separation, reflection involution, off-center anatomy, oblique grids, and native-grid round trips.
6. Change the crop schema version and reject earlier bundles rather than silently migrating them.

CT is linearly resampled to PET. Masks use nearest-neighbor interpolation. PET SUVbw remains the quantitative reference grid.

## Gate 3: simulation and transparent baseline

The simulator must retain sham identity, fractional-volume accuracy, activity conservation, recovered blur width, deterministic seeds, and complete provenance. Gaussian blur remains a controlled image-domain factor.

If CT-input registration stress remains in scope, the pipeline must apply it only to CT while PET and labels remain fixed. No result is reported until that path is tested. Acquisition-noise claims remain out of scope unless a validated image-formation model is added.

The threshold baseline must implement the same final definitions used for learned-model comparison:

- 26-connected 3D components.
- Minimum physical component volume.
- Frozen source intersection and tie rules.
- Sham and untouched validation activation in the F2 selection set.
- One validation-only threshold and volume freeze.
- Explicit infeasible status when the untouched-component ceiling cannot be met.

## Gate 4: native-space learned evaluation

The B2 model remains exploratory until the following evaluation is complete:

1. Run deterministic center-slice inference across the eligible crop.
2. Stitch scores into the complete crop without missing or duplicated slices.
3. Map the result back to original PET shape and affine.
4. Apply the frozen score threshold and physical component-volume rule.
5. Match one synthetic source by intersection under the frozen tie order.
6. Calculate subject-level and condition-level summaries without averaging away misses.
7. Use 5,000 subject-cluster bootstrap replicates and keep every condition for each resampled subject together.
8. Freeze the model and operating point on validation, then access the sealed test subjects once.

The report will show raw subject counts beside intervals. Center slices and synthetic conditions are never counted as independent subjects.

## Gate 5: deterministic quantification

Both ground-truth and predicted masks pass through the same complete 3D quantifier. The evaluation reports error in SUVmax, SUVmean, physical mask volume, longitudinal extent, contralateral ratio, and laterality. Empty or invalid values remain structured nulls with typed reasons.

The product layer must call this quantifier using raw SUVbw, original geometry, the frozen reflection transform, and native-space masks. Pixel-count volume proxies, clipped contralateral values, approximate array-midpoint laterality, and forced single-slice extent are not accepted as final measurements.

## Gate 6: product orchestration and evidence retrieval

The product keeps deterministic measurement separate from optional prose generation. A report generator may organize interpretation and limitations, but every number is copied from structured code output and checked by a deterministic verifier.

The default reference backend is labeled as an integration fixture that returns a known synthetic mask. The learned backend must be labeled as exploratory until complete-scan evaluation is integrated.

The publication-safe retrieval index will be rebuilt only from approved public sources. Its evaluation records corpus hashes, query definitions, relevance labels, recall at k, reranked ranking metrics, and failure cases. Earlier retrieval scores are not carried forward.

## Gate 7: report build and editorial checks

The report build sequence is:

```bash
python3 docs/report/scripts/build_report_figures.py
cd docs/report
pdflatex -interaction=nonstopmode -halt-on-error VascuTrace_Technical_Report_2026-07-20.tex
bibtex VascuTrace_Technical_Report_2026-07-20
pdflatex -interaction=nonstopmode -halt-on-error VascuTrace_Technical_Report_2026-07-20.tex
pdflatex -interaction=nonstopmode -halt-on-error VascuTrace_Technical_Report_2026-07-20.tex
```

Acceptance requires:

- Eight readable figures generated only from aggregate values or synthetic schematics.
- Complete references with no undefined citation or cross-reference.
- No missing graphic or material overfull box.
- Embedded fonts and readable PDF metadata.
- Exact warning text present.
- No Unicode em dash or literal triple-hyphen artifact in source or extracted PDF text.
- No unfinished drafting marker, private development reference, identifier, or local absolute path.
- Independent scientific, numeric-fidelity, citation, editorial, and full-page visual checks.

## Gate 8: software verification

Run from a clean process:

```bash
uv run ruff check --no-cache .
uv run ruff format --check --no-cache .
uv run pytest -q -m 'not local_data and not gpu'
```

Run targeted geometry, simulation, baseline, inference, quantification, product, report-verifier, and MCP tests after any related change. Data and GPU suites remain separate and must record their environment and exit codes.

If the multi-worker data-loader test hangs in a constrained execution environment, run that test independently and run the rest of the offline suite with an explicit exclusion. Report both commands. Do not describe a partial run as a full-suite pass.

## Public snapshot review

Build the release from an explicit file manifest. Review every candidate text file and every reachable development-branch object for:

- private development configuration or instructions;
- credentials, local paths, or runtime state;
- data identifiers or per-case measurements;
- model weights, caches, and generated run outputs;
- contaminated retrieval material;
- files larger than 25 MB;
- prohibited scientific wording.

The development branch may be rewritten to a reviewed no-parent public snapshot if that is required to make excluded material unreachable from the branch. The release report should state only what can be proven: excluded material is no longer reachable from the published development branch. It should not claim immediate physical destruction of unreachable objects on the hosting service.

## Release acceptance matrix

| Area | Required evidence | Stop condition |
|:--|:--|:--|
| Science | Exact warning, bounded claims, independent review | Unsupported clinical or scanner claim |
| Data | Current archive identity, aggregate-only public artifacts | Identifier, medical image, or case row present |
| Geometry | Physical transforms and reconciled reflection | Legacy reflection used for a promoted result |
| Baseline | Promotion-compliant frozen baseline | Mismatched negative or volume definition |
| Model | Native-space 3D validation and sealed test | Center-slice validation presented as a test result |
| Quantification | Raw-SUV 3D measurements and structured nulls | 2D adapter output presented as complete physical measurement |
| Retrieval | Public-only rebuilt corpus and fresh evaluation | Earlier excluded index or score included |
| Report | Clean compile, complete visual inspection, editorial gates | Missing citation, unreadable figure, or drafting artifact |
| Code | Lint, format, tests, run-and-observe path | Unresolved high-severity failure |
| Publication | Explicit manifest and clean reachable history | Excluded development or data material remains reachable |

## Completion definition

The publication is complete when the PDF and source agree, the eight figures are readable, evidence is traceable, scientific boundaries are intact, software checks pass, the public manifest contains only approved artifacts, the sanitized development branch is pushed with its exact commit recorded, and a post-push scan confirms the remote branch tree and reachable history.
