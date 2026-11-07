import copy
from pathlib import Path

import pytest
import yaml

from common.cerberus_policies import backend_card
from common.cerberus_policies.backend_card import CardValidationError, load_card, validate_card

GOOD_L0 = {
    "name": "some_backend",
    "slot": "loco",
    "boundary": "B1_LOCO",
    "paradigm": "A. blind",
    "architecture": "modular",
    "required_sensors": "unknown",
    "rate_hz": "unknown",
    "license": "unknown",
    "checkpoint_url": "unknown",
    "code_url": "unknown",
    "support_level": "L0",
    "target_support_level": "L2",
    "evidence": [],
    "results": [],
}
EV_FILE = {"kind": "file", "ref": "common/cerberus_policies/base.py"}
EV_REPORT = {"kind": "report", "ref": "docs/h1_report.md"}
RESULT = {"metric": "success", "value": 0.9, "eval": "push_recovery", "source": "docs/h1_report.md"}


def good_l1():
    c = copy.deepcopy(GOOD_L0)
    c.update(support_level="L1", required_sensors=["imu"], rate_hz=50, license="MIT", checkpoint_url="n/a",
             code_url="common/x.py", evidence=[EV_FILE])
    return c


def problems(card, **kw):
    with pytest.raises(CardValidationError) as e:
        validate_card(card, **kw)
    return " | ".join(e.value.problems)


def test_good_cards_validate():
    assert validate_card(GOOD_L0).support_level == "L0"
    assert validate_card(good_l1(), check_paths=True).name == "some_backend"


def test_l0_may_have_empty_evidence_and_unknowns():
    validate_card(GOOD_L0)


def test_l1_without_evidence_fails():
    c = good_l1()
    c["evidence"] = []
    assert "requires non-empty evidence" in problems(c)


def test_l2_without_results_fails():
    c = good_l1()
    c.update(support_level="L2", target_support_level="L2", evidence=[EV_FILE, EV_REPORT])
    assert "requires a results entry" in problems(c)
    c["results"] = [RESULT]
    validate_card(c)


def test_l2_needs_report_evidence_kind():
    c = good_l1()
    c.update(support_level="L2", target_support_level="L2", results=[RESULT])
    assert "'report' evidence" in problems(c)


def test_l3_needs_trained_by_us_result():
    c = good_l1()
    c.update(support_level="L3", target_support_level="L3", evidence=[EV_FILE, EV_REPORT], results=[RESULT])
    assert "trained_by_us" in problems(c)
    c["results"] = [{**RESULT, "trained_by_us": True}]
    validate_card(c)


def test_unknown_only_allowed_at_l0():
    c = good_l1()
    c["license"] = "unknown"
    assert "only at L0" in problems(c)


def test_target_must_not_be_below_support():
    c = good_l1()
    c["target_support_level"] = "L0"
    assert "target_support_level must be >=" in problems(c)


def test_missing_and_extra_fields():
    c = copy.deepcopy(GOOD_L0)
    del c["paradigm"]
    assert "missing fields" in problems(c)
    c = copy.deepcopy(GOOD_L0)
    c["extra"] = 1
    assert "unknown fields" in problems(c)


def test_enum_and_consistency_errors():
    c = copy.deepcopy(GOOD_L0)
    c.update(slot="legs", architecture="hybrid", support_level="L9")
    msg = problems(c)
    assert "slot must be" in msg and "architecture must be" in msg and "support_level must be" in msg
    c = copy.deepcopy(GOOD_L0)
    c["boundary"] = "B2_ARM"
    assert "does not match slot" in problems(c)


def test_evidence_file_must_exist_when_paths_checked():
    c = good_l1()
    c["evidence"] = [{"kind": "file", "ref": "no/such/file.py"}]
    validate_card(c)  # path existence is only checked on request
    assert "does not exist" in problems(c, check_paths=True)


def test_bad_evidence_entries():
    c = good_l1()
    c["evidence"] = ["common/x.py"]
    assert "evidence[0]" in problems(c)


def test_load_card_name_must_match_file(tmp_path: Path):
    p = tmp_path / "other_name.yaml"
    p.write_text(yaml.safe_dump(GOOD_L0))
    with pytest.raises(CardValidationError, match="must match file name"):
        load_card(p)


def test_schema_file_enums_match_validator():
    schema = yaml.safe_load((backend_card.REPO_ROOT / "docs/backend_card.schema.yaml").read_text())
    props = schema["properties"]
    assert tuple(schema["required"]) == backend_card.REQUIRED_FIELDS
    assert tuple(props["slot"]["enum"]) == backend_card.SLOTS
    assert tuple(props["boundary"]["enum"]) == backend_card.BOUNDARIES
    assert tuple(props["architecture"]["enum"]) == backend_card.ARCHITECTURES
    assert tuple(props["support_level"]["enum"]) == backend_card.SUPPORT_LEVELS
    assert tuple(props["target_support_level"]["enum"]) == backend_card.SUPPORT_LEVELS
    assert tuple(props["evidence"]["items"]["properties"]["kind"]["enum"]) == backend_card.EVIDENCE_KINDS
