"""docs/backend_support.md must be regenerated whenever a backend card changes."""

import gen_backend_table
from common.cerberus_policies.backend_card import load_all_cards


def test_backend_support_md_is_up_to_date():
    expected = gen_backend_table.render()
    actual = gen_backend_table.OUTPUT.read_text(encoding="utf-8")
    assert actual == expected, "docs/backend_support.md is stale: run `pixi run backend-table`"


def test_every_card_appears_in_the_table():
    text = gen_backend_table.render()
    for card in load_all_cards():
        assert f"`{card.name}`" in text


def test_render_is_deterministic_and_escapes_pipes():
    assert gen_backend_table.render() == gen_backend_table.render()
    assert gen_backend_table._cell("a|b") == "a\\|b"
