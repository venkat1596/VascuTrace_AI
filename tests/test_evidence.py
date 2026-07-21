import json
from pathlib import Path

from vascutrace.evidence import EvidenceStore, is_cache_eligible
from vascutrace.orchestrator import run_evidence_request


def test_corpus_has_complete_provenance() -> None:
    store = EvidenceStore()
    assert len(store.documents) >= 4
    for document in store.documents:
        assert document.document_id
        assert document.title
        assert document.source_url.startswith("https://")
        assert document.published_year
        assert document.license


def test_general_explanation_uses_semantic_cache() -> None:
    store = EvidenceStore(similarity_threshold=0.75)
    first = store.search("What is the partial volume effect in PET?", top_k=2)
    second = store.search("Explain partial volume effect in PET", top_k=2)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.evidence == second.evidence
    assert first.evidence[0].citation_id == "soret_partial_volume_2007"


def test_case_measurements_never_enter_cache() -> None:
    store = EvidenceStore()
    first = store.search("What is SUVmax for subject 006?")
    second = store.search("What is SUVmax for subject 006?")

    assert not first.cache_eligible
    assert not second.cache_hit
    assert store.cache_size == 0
    assert not is_cache_eligible("Which side is abnormal in this scan?")
    assert is_cache_eligible("What does SUVmax mean?")


def test_ingestion_rejects_incomplete_provenance(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"document_id": "missing-fields"}]))

    try:
        EvidenceStore(corpus)
    except ValueError as error:
        assert "title" in str(error)
    else:
        raise AssertionError("Incomplete provenance was accepted")


def test_evidence_orchestrator_route() -> None:
    result = run_evidence_request("Why review PET CT misregistration?", top_k=1)
    assert result.route == "evidence"
    assert result.trace == ["retrieve_research_evidence"]
    assert result.payload["evidence"][0]["citation_id"] == (
        "blodgett_pet_ct_misregistration_2005"
    )
