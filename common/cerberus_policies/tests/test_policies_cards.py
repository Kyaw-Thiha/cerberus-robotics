from common.cerberus_policies.backend_card import CARDS_DIR, load_all_cards

EXPECTED_L0 = {"blind_ppo", "jeplo", "dawn", "height_map_miki", "extreme_parkour", "omni_perception"}
EXPECTED_TARGETS = {
    "blind_ppo": "L3", "jeplo": "L2", "dawn": "L1", "height_map_miki": "L1",
    "extreme_parkour": "L0", "omni_perception": "L1", "null_loco": "L1", "scripted_loco": "L1",
}


def test_every_card_validates_including_evidence_paths():
    cards = load_all_cards(CARDS_DIR, check_paths=True)
    assert len(cards) == len(list(CARDS_DIR.glob("*.yaml"))) >= 8


def test_honest_levels_and_targets():
    cards = {c.name: c for c in load_all_cards(CARDS_DIR)}
    assert {n for n, c in cards.items() if c.support_level == "L0"} == EXPECTED_L0
    assert cards["null_loco"].support_level == cards["scripted_loco"].support_level == "L1"
    assert {n: c.target_support_level for n, c in cards.items()} == EXPECTED_TARGETS
    # no L2/L3 claim exists yet, nothing was trained or evaluated by this handoff
    assert all(c.support_level in ("L0", "L1") for c in cards.values())


def test_fused_cards_declare_raw_streams():
    cards = {c.name: c for c in load_all_cards(CARDS_DIR)}
    for name in ("jeplo", "dawn", "omni_perception", "extreme_parkour"):
        assert cards[name].architecture == "fused"
        assert cards[name].required_sensors not in ([], "unknown")
