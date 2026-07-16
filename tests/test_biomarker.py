"""Tests for the biomarker distant-supervision labeler (the fine-tune mechanism)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import biomarker as B
from bpe import BPETokenizer

TOK = BPETokenizer.load(Path(__file__).resolve().parents[1] / "pilot" / "tokenizers" / "domain_4096.json")


def test_find_spans_hits_biomarkers():
    t = "EGFR T790M mutation and ALK rearrangement; PD-L1 expression high, MSI-High, TMB."
    got = {t[s:e] for s, e in B.find_spans(t)}
    assert "EGFR" in got
    assert any("T790M" in g for g in got)
    assert any("ALK" in g for g in got)
    assert any("PD-L1" in g for g in got)
    assert "MSI-High" in got


def test_find_spans_clean_on_plain_text():
    assert B.find_spans("No prior systemic therapy for advanced disease is allowed.") == []


def test_label_roundtrip_and_alignment():
    text = "Patients with BRAF V600E mutated melanoma; exclude KRAS G12C."
    ids, labs = B.label_tokens(TOK, text)
    # ids reconstruct the exact text (byte-lossless)
    assert b"".join(TOK.vocab[i] for i in ids).decode("utf-8") == text
    assert len(labs) == len(ids)
    assert sum(labs) > 0
    # tokens overlapping "BRAF" must be labeled 1
    tagged = "".join(TOK.vocab[i].decode("utf-8", "replace") for i, l in zip(ids, labs) if l)
    assert "BRAF" in tagged and "V600E" in tagged and "KRAS" in tagged


def test_label_ids_capping_robust():
    text = "HER2-positive breast cancer with PIK3CA mutation and PTEN loss."
    ids, _ = B.label_tokens(TOK, text)
    sub = ids[4:]                                   # simulate context-capping (drop leading tokens)
    labs = B.label_ids(TOK, sub)
    assert len(labs) == len(sub)                    # stays aligned to the given ids


def test_plain_text_has_no_labels():
    ids, labs = B.label_tokens(TOK, "The study will enroll adults aged 18 years or older.")
    assert sum(labs) == 0
