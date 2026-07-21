from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPOSITORY_ROOT / "docs" / "report" / "scripts" / "build_codex_session_evidence.py"
)
EVIDENCE_FILENAME = "codex_session_evidence.json"
TIMELINE_FILENAME = "12_codex_session_timeline.png"
ACTIVITY_FILENAME = "13_codex_session_activity.png"
ARTIFACT_FILENAMES = (
    EVIDENCE_FILENAME,
    TIMELINE_FILENAME,
    ACTIVITY_FILENAME,
)
WORKSPACE_NAME = "SyntheticResearchWorkspace"
CUTOFF_UTC = "2026-01-02T00:00:00Z"
COUNT_KEYS = (
    "user_turns",
    "assistant_updates",
    "tasks",
    "tool_calls",
    "patch_events",
    "web_searches",
    "bounded_review_activities",
    "compactions",
)
EXPECTED_DEFAULT_IDS = (
    "019f5c4e-1f8d-7190-8a04-2c6f7e7a2e18",
    "019f5c9a-0be0-7351-b9c1-9d7b342d8108",
    "019f6015-b646-70b3-ab11-6d5b20d9eeee",
    "019f60c0-ba06-77f0-be21-37da29651799",
    "019f6187-81f0-7db1-b5c1-d9cdbba313dc",
    "019f661f-b909-7a22-b165-7d5c8e5da35f",
    "019f6773-1bc8-73d2-9a39-b7ade344260f",
    "019f6bd0-46c4-7121-bf6d-88481daf4566",
    "019f6d22-f305-7ca0-86ae-20e1ac55dba6",
    "019f7398-2350-79d0-a657-fca01373af04",
    "019f81d0-b3be-7e21-a19a-491a13e360e8",
)

RAW_USER_CANARY = "PRIVATE_RAW_USER_MESSAGE_CANARY"
RAW_ASSISTANT_CANARY = "PRIVATE_RAW_ASSISTANT_MESSAGE_CANARY"
REASONING_CANARY = "PRIVATE_REASONING_CANARY"
SYSTEM_CANARY = "PRIVATE_SYSTEM_MESSAGE_CANARY"
DEVELOPER_CANARY = "PRIVATE_DEVELOPER_MESSAGE_CANARY"
TOOL_ARGUMENT_CANARY = "PRIVATE_TOOL_ARGUMENT_CANARY"
TOOL_OUTPUT_CANARY = "PRIVATE_TOOL_OUTPUT_CANARY"
INSTRUCTIONS_CANARY = "PRIVATE_INSTRUCTIONS_CANARY"
SYNTHETIC_ABSOLUTE_PATH = "/synthetic/private/workspaces/SyntheticResearchWorkspace"
PRIVATE_CANARIES = (
    RAW_USER_CANARY,
    RAW_ASSISTANT_CANARY,
    REASONING_CANARY,
    SYSTEM_CANARY,
    DEVELOPER_CANARY,
    TOOL_ARGUMENT_CANARY,
    TOOL_OUTPUT_CANARY,
    INSTRUCTIONS_CANARY,
    SYNTHETIC_ABSOLUTE_PATH,
)


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "codex_session_evidence_builder", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load builder specification: {SCRIPT_PATH.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _synthetic_session_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


def _timestamp(offset_seconds: int) -> str:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = start + timedelta(seconds=offset_seconds)
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _record(
    record_type: str, offset_seconds: int, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp(offset_seconds),
        "type": record_type,
        "payload": payload,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        f"{json.dumps(record, sort_keys=True, separators=(',', ':'))}\n"
        for record in records
    )
    path.write_text(serialized, encoding="utf-8")


def _session_records(
    index: int,
    session_id: str,
    *,
    workspace_name: str = WORKSPACE_NAME,
    metadata_session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    user_turns = index % 3 + 1
    assistant_updates = index % 4 + 1
    tasks = index % 2 + 1
    function_calls = index % 2 + 1
    custom_tool_calls = 1
    mcp_tool_calls = 1 if index % 3 == 0 else 0
    patch_events = index % 3
    web_searches = index % 2
    bounded_review_activities = (index + 1) % 3
    compactions = index % 2
    expected_counts = {
        "user_turns": user_turns,
        "assistant_updates": assistant_updates,
        "tasks": tasks,
        "tool_calls": function_calls + custom_tool_calls + mcp_tool_calls,
        "patch_events": patch_events,
        "web_searches": web_searches,
        "bounded_review_activities": bounded_review_activities,
        "compactions": compactions,
    }

    offset = index * 1000
    records: list[dict[str, Any]] = [
        _record(
            "session_meta",
            offset,
            {
                "id": metadata_session_id or session_id,
                "timestamp": _timestamp(offset),
                "cwd": f"/synthetic/private/workspaces/{workspace_name}",
                "source": "vscode",
                "originator": "synthetic-test-client",
                "cli_version": "0.0-test",
                "base_instructions": INSTRUCTIONS_CANARY,
            },
        ),
        _record(
            "turn_context",
            offset + 1,
            {
                "model": "gpt-5.6-sol",
                "cwd": f"/synthetic/private/workspaces/{workspace_name}",
                "user_instructions": INSTRUCTIONS_CANARY,
            },
        ),
    ]
    if index == 0:
        records.append(
            _record(
                "turn_context",
                offset + 2,
                {
                    "model": "codex-auto-review",
                    "cwd": SYNTHETIC_ABSOLUTE_PATH,
                    "user_instructions": INSTRUCTIONS_CANARY,
                },
            )
        )

    next_offset = offset + 10

    def append(record_type: str, payload: dict[str, Any]) -> None:
        nonlocal next_offset
        records.append(_record(record_type, next_offset, payload))
        next_offset += 1

    for _ in range(user_turns):
        append("event_msg", {"type": "user_message", "message": RAW_USER_CANARY})
    for _ in range(assistant_updates):
        append(
            "event_msg",
            {"type": "agent_message", "message": RAW_ASSISTANT_CANARY},
        )
    for _ in range(tasks):
        append("event_msg", {"type": "task_started", "detail": "synthetic"})
    for _ in range(function_calls):
        append(
            "response_item",
            {
                "type": "function_call",
                "name": "synthetic_function",
                "arguments": TOOL_ARGUMENT_CANARY,
            },
        )
        append(
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "synthetic-call",
                "output": TOOL_OUTPUT_CANARY,
            },
        )
    for _ in range(custom_tool_calls):
        append(
            "response_item",
            {
                "type": "custom_tool_call",
                "name": "synthetic_custom_tool",
                "input": TOOL_ARGUMENT_CANARY,
            },
        )
        append(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "call_id": "synthetic-custom-call",
                "output": TOOL_OUTPUT_CANARY,
            },
        )
    for _ in range(mcp_tool_calls):
        append(
            "response_item",
            {
                "type": "mcp_tool_call",
                "name": "synthetic_mcp_tool",
                "arguments": TOOL_ARGUMENT_CANARY,
            },
        )
        append(
            "response_item",
            {
                "type": "mcp_tool_call_output",
                "output": TOOL_OUTPUT_CANARY,
            },
        )
    for _ in range(patch_events):
        append("event_msg", {"type": "patch_apply_end", "status": "success"})
    for _ in range(web_searches):
        append("event_msg", {"type": "web_search_end", "status": "success"})
    for _ in range(bounded_review_activities):
        append("event_msg", {"type": "sub_agent_activity", "status": "complete"})
    for _ in range(compactions):
        append("compacted", {"replacement": "synthetic summary"})

    append(
        "response_item",
        {"type": "reasoning", "summary": [{"text": REASONING_CANARY}]},
    )
    append(
        "response_item",
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_CANARY}],
        },
    )
    append(
        "response_item",
        {
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": DEVELOPER_CANARY}],
        },
    )
    append(
        "response_item",
        {
            "type": "function_call_output",
            "call_id": "unpaired-private-output",
            "output": TOOL_OUTPUT_CANARY,
        },
    )
    return records, expected_counts


def _write_session(
    root: Path,
    index: int,
    session_id: str,
    *,
    workspace_name: str = WORKSPACE_NAME,
    metadata_session_id: str | None = None,
    branch: str = "main",
) -> tuple[Path, dict[str, int]]:
    path = root / branch / f"rollout-2026-01-01T00-00-{index:02d}-{session_id}.jsonl"
    records, counts = _session_records(
        index,
        session_id,
        workspace_name=workspace_name,
        metadata_session_id=metadata_session_id,
    )
    _write_jsonl(path, records)
    return path, counts


def _make_specs() -> tuple[Any, ...]:
    return tuple(
        builder.SessionSpec(
            session_id=_synthetic_session_id(index),
            primary=index == 4,
            task_themes=(f"Synthetic task theme {index + 1}",),
            public_outcomes=(f"docs/synthetic-outcome-{index + 1}.md",),
        )
        for index in range(11)
    )


def _make_valid_source(root: Path) -> tuple[tuple[Any, ...], dict[str, dict[str, int]]]:
    specs = _make_specs()
    expected: dict[str, dict[str, int]] = {}
    for index, session_spec in enumerate(specs):
        _, counts = _write_session(root, index, session_spec.session_id)
        expected[session_spec.session_id] = counts
    return specs, expected


def _build(root: Path, output_root: Path, specs: tuple[Any, ...]) -> dict[str, Any]:
    builder.build_evidence(
        session_root=root,
        output_root=output_root,
        cutoff_utc=CUTOFF_UTC,
        session_specs=specs,
        workspace_name=WORKSPACE_NAME,
    )
    return json.loads((output_root / EVIDENCE_FILENAME).read_text(encoding="utf-8"))


def _artifact_bytes(output_root: Path) -> dict[str, bytes]:
    return {
        filename: (output_root / filename).read_bytes()
        for filename in ARTIFACT_FILENAMES
    }


def _assert_valid_png(path: Path) -> None:
    content = path.read_bytes()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    assert content[12:16] == b"IHDR"
    width, height = struct.unpack(">II", content[16:24])
    assert width >= 1200
    assert height >= 700


def _all_mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for item in value.values():
            keys.update(_all_mapping_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_all_mapping_keys(item))
        return keys
    return set()


def test_public_api_and_cli_defaults_are_explicit() -> None:
    assert callable(builder.build_evidence)
    assert builder.SessionSpec is not None
    default_specs = tuple(builder.DEFAULT_SESSION_SPECS)
    assert tuple(spec.session_id for spec in default_specs) == EXPECTED_DEFAULT_IDS
    assert sum(bool(spec.primary) for spec in default_specs) == 1

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--session-root" in result.stdout
    assert "--output-root" in result.stdout
    assert "--cutoff" in result.stdout


def test_builds_exact_allowlist_rows_and_aggregate_sums(tmp_path: Path) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs, expected = _make_valid_source(source_root)

    receipt = _build(source_root, output_root, specs)

    assert receipt["schema_version"] == 1
    assert receipt["workspace_name"] == WORKSPACE_NAME
    assert receipt["cutoff_utc"] == CUTOFF_UTC
    assert receipt["session_count"] == 11
    assert receipt["primary_session_id"] == specs[4].session_id
    assert [row["session_id"] for row in receipt["sessions"]] == [
        spec.session_id for spec in specs
    ]
    assert sum(bool(row["primary"]) for row in receipt["sessions"]) == 1

    for row, spec in zip(receipt["sessions"], specs, strict=True):
        assert row["counts"] == expected[spec.session_id]
        assert row["task_themes"] == list(spec.task_themes)
        assert row["public_outcomes"] == list(spec.public_outcomes)
        assert row["recorded_surfaces"] == ["vscode"]
        assert row["start_utc"].endswith("Z")
        assert row["end_utc"].endswith("Z")

    for count_key in COUNT_KEYS:
        row_sum = sum(row["counts"][count_key] for row in receipt["sessions"])
        expected_sum = sum(counts[count_key] for counts in expected.values())
        assert receipt["aggregates"][count_key] == row_sum == expected_sum

    assert receipt["aggregates"]["session_count"] == 11
    _assert_valid_png(output_root / TIMELINE_FILENAME)
    _assert_valid_png(output_root / ACTIVITY_FILENAME)


def test_records_exact_observed_models_and_excludes_unrelated_session(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs, _ = _make_valid_source(source_root)
    unrelated_id = _synthetic_session_id(99)
    _write_session(source_root, 99, unrelated_id, branch="unrelated")

    receipt = _build(source_root, output_root, specs)

    assert unrelated_id not in json.dumps(receipt, sort_keys=True)
    assert receipt["sessions"][0]["recorded_model_identifiers"] == [
        "codex-auto-review",
        "gpt-5.6-sol",
    ]
    for row in receipt["sessions"][1:]:
        assert row["recorded_model_identifiers"] == ["gpt-5.6-sol"]
    assert "unverified" not in json.dumps(receipt).lower()


def test_cutoff_makes_json_and_figures_byte_deterministic(tmp_path: Path) -> None:
    source_root = tmp_path / "sessions"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    specs, _ = _make_valid_source(source_root)

    _build(source_root, first_output, specs)
    first_bytes = _artifact_bytes(first_output)

    active_file = next(source_root.rglob(f"*{specs[-1].session_id}.jsonl"))
    with active_file.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp": "2026-01-03T00:00:00Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "user_message",
                        "message": "POST_CUTOFF_CANARY",
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        stream.write("\n")

    _build(source_root, second_output, specs)
    second_bytes = _artifact_bytes(second_output)

    assert first_bytes == second_bytes
    assert b"POST_CUTOFF_CANARY" not in second_bytes[EVIDENCE_FILENAME]


def test_private_payload_lanes_and_absolute_paths_never_escape(tmp_path: Path) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs, _ = _make_valid_source(source_root)

    receipt = _build(source_root, output_root, specs)
    artifacts = _artifact_bytes(output_root)
    combined = b"\n".join(artifacts.values())

    for canary in PRIVATE_CANARIES:
        assert canary.encode("utf-8") not in combined
    assert b"/synthetic/" not in combined
    assert b"POST_CUTOFF_CANARY" not in combined

    forbidden_keys = {
        "arguments",
        "content",
        "cwd",
        "input",
        "instructions",
        "message",
        "output",
        "reasoning",
        "source_path",
        "transcript",
    }
    assert _all_mapping_keys(receipt).isdisjoint(forbidden_keys)


def test_missing_allowlisted_session_fails_without_partial_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs = _make_specs()
    for index, session_spec in enumerate(specs[:-1]):
        _write_session(source_root, index, session_spec.session_id)

    with pytest.raises(ValueError, match=r"(?i)missing"):
        _build(source_root, output_root, specs)

    assert not any((output_root / name).exists() for name in ARTIFACT_FILENAMES)


def test_duplicate_allowlisted_session_fails_without_partial_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs, _ = _make_valid_source(source_root)
    original = next(source_root.rglob(f"*{specs[0].session_id}.jsonl"))
    duplicate = source_root / "duplicate" / original.name
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(original, duplicate)

    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        _build(source_root, output_root, specs)

    assert not any((output_root / name).exists() for name in ARTIFACT_FILENAMES)


def test_filename_and_session_metadata_mismatch_fails_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs = _make_specs()
    for index, session_spec in enumerate(specs):
        metadata_id = _synthetic_session_id(88) if index == 3 else None
        _write_session(
            source_root,
            index,
            session_spec.session_id,
            metadata_session_id=metadata_id,
        )

    with pytest.raises(ValueError, match=r"(?i)(mismatch|does not match)"):
        _build(source_root, output_root, specs)

    assert not any((output_root / name).exists() for name in ARTIFACT_FILENAMES)


def test_workspace_mismatch_fails_closed(tmp_path: Path) -> None:
    source_root = tmp_path / "sessions"
    output_root = tmp_path / "public"
    specs = _make_specs()
    for index, session_spec in enumerate(specs):
        workspace_name = "DifferentSyntheticWorkspace" if index == 7 else WORKSPACE_NAME
        _write_session(
            source_root,
            index,
            session_spec.session_id,
            workspace_name=workspace_name,
        )

    with pytest.raises(ValueError, match=r"(?i)workspace"):
        _build(source_root, output_root, specs)

    assert not any((output_root / name).exists() for name in ARTIFACT_FILENAMES)


def test_test_source_has_no_forbidden_typographic_artifacts() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert "\N{EM DASH}" not in source
    assert "-" * 3 not in source
