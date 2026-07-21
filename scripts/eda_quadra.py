"""Reproducible cohort, geometry, intensity, and anatomy-anchor EDA for QUADRA_HC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def image_row(path: Path, root: Path, load_intensity: bool) -> dict:
    image = nib.load(path)
    subject = next(part for part in path.parts if part.startswith("QUADRA_HC_"))
    session = "Retest" if "Retest" in path.parts else "Test"
    modality = "PET" if "PT-SUV" in path.name else "CT"
    row = {
        "subject": subject,
        "session": session,
        "modality": modality,
        "path": str(path.relative_to(root)),
        "shape": "x".join(map(str, image.shape)),
        "spacing_mm": "x".join(
            f"{value:.6g}" for value in image.header.get_zooms()[:3]
        ),
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "dtype": str(image.get_data_dtype()),
        "size_bytes": path.stat().st_size,
    }
    if load_intensity:
        data = np.asanyarray(image.dataobj)
        row.update(
            finite=bool(np.isfinite(data).all()),
            minimum=float(np.nanmin(data)),
            median=float(np.nanmedian(data)),
            p99=float(np.nanpercentile(data, 99)),
            maximum=float(np.nanmax(data)),
        )
    return row


def anchor_row(path: Path) -> dict:
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    subject = next(part for part in path.parts if part.startswith("QUADRA_HC_"))
    session = "Retest" if "Retest" in path.parts else "Test"
    labels = (7, 8) if "Cardiac" in path.name else (5, 6)
    names = (
        ("iliac_artery_left", "iliac_artery_right")
        if labels[0] == 7
        else (
            "femur_left",
            "femur_right",
        )
    )
    row = {"subject": subject, "session": session}
    for label, name in zip(labels, names, strict=True):
        indices = np.argwhere(data == label)
        row[f"{name}_voxels"] = len(indices)
        row[f"{name}_present"] = bool(len(indices))
        if len(indices):
            row[f"{name}_z_min"] = int(indices[:, 2].min())
            row[f"{name}_z_max"] = int(indices[:, 2].max())
    return row


def demographics_summary(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_excel(path, header=[0, 1])
    test_weight, retest_weight = frame.iloc[:, 4], frame.iloc[:, 8]
    test_bmi, retest_bmi = frame.iloc[:, 5], frame.iloc[:, 9]
    test_activity, retest_activity = frame.iloc[:, 7], frame.iloc[:, 11]
    activity_delta = retest_activity - test_activity
    summary = {
        "rows": len(frame),
        "missing_values": int(frame.isna().sum().sum()),
        "sex_counts": frame.iloc[:, 1].value_counts().to_dict(),
        "age_years": frame.iloc[:, 3].describe().to_dict(),
        "test_bmi": test_bmi.describe().to_dict(),
        "scan_interval_days": frame.iloc[:, 14].describe().to_dict(),
        "weight_test_retest_correlation": float(test_weight.corr(retest_weight)),
        "bmi_test_retest_correlation": float(test_bmi.corr(retest_bmi)),
        "activity_delta_mbq": activity_delta.describe().to_dict(),
        "activity_delta_over_20_mbq_subjects": frame.loc[
            activity_delta.abs() > 20, frame.columns[0]
        ]
        .astype(int)
        .tolist(),
        "scan_interval_outlier_subjects": frame.loc[
            (frame.iloc[:, 14] < 28) | (frame.iloc[:, 14] > 50), frame.columns[0]
        ]
        .astype(int)
        .tolist(),
    }
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eda/quadra_hc")
    )
    parser.add_argument("--intensity-subjects", type=int, default=4)
    args = parser.parse_args()
    root = args.dataset_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    pet_paths = sorted(root.rglob("*PT-SUV.nii.gz"))
    ct_paths = sorted(root.rglob("*CT-AC.nii.gz"))
    selected = {
        f"QUADRA_HC_{index:03d}" for index in range(1, args.intensity_subjects + 1)
    }
    images = pd.DataFrame(
        image_row(path, root, any(subject in path.parts for subject in selected))
        for path in [*pet_paths, *ct_paths]
    )
    images.to_csv(output / "image_inventory.csv", index=False)

    anchor_paths = sorted(root.rglob("*Cardiac.nii.gz")) + sorted(
        root.rglob("*Peripheral-Bones.nii.gz")
    )
    anchors = pd.DataFrame(anchor_row(path) for path in anchor_paths)
    anchors = anchors.groupby(["subject", "session"], as_index=False).first()
    anchors.to_csv(output / "anatomy_anchor_inventory.csv", index=False)

    demographics, demographic_stats = demographics_summary(
        next(root.rglob("Demographics.xlsx"))
    )
    demographics.to_csv(output / "demographics.csv", index=False)
    segmentation_count = len(list(root.rglob("*/Segmentations/*.nii.gz")))
    summary = {
        "subjects": images.subject.nunique(),
        "sessions": images.session.value_counts().to_dict(),
        "pet_files": len(pet_paths),
        "ct_files": len(ct_paths),
        "segmentation_files": segmentation_count,
        "geometry_counts": images.groupby(
            ["modality", "shape", "spacing_mm", "orientation"]
        )
        .size()
        .reset_index(name=chr(99) + chr(111) + chr(117) + chr(110) + chr(116))
        .to_dict(
            orient=chr(114)
            + chr(101)
            + chr(99)
            + chr(111)
            + chr(114)
            + chr(100)
            + chr(115)
        ),
        "missing_iliac_left": int((anchors.iliac_artery_left_present.eq(False)).sum()),
        "missing_iliac_right": int(
            (anchors.iliac_artery_right_present.eq(False)).sum()
        ),
        "missing_femur_left": int((anchors.femur_left_present.eq(False)).sum()),
        "missing_femur_right": int((anchors.femur_right_present.eq(False)).sum()),
        "demographics": demographic_stats,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: float(value)) + "\n"
    )
    print(json.dumps(summary, indent=2, default=lambda value: float(value)))


if __name__ == "__main__":
    main()
