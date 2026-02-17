from __future__ import annotations

from app.domain.construction import merge_top_down_filters


def test_merge_top_down_filters_overrides_with_context():
    base = {"project_id": "P1", "work_type": "earthwork", "custom": "x"}
    context = {"project_id": "P2", "building": "TOWER-A"}
    merged = merge_top_down_filters(base, context)
    assert merged["project_id"] == "P2"
    assert merged["work_type"] == "earthwork"
    assert merged["building"] == "TOWER-A"
    assert merged["custom"] == "x"
