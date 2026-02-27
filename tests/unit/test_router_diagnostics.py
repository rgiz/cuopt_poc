from src.plan.models import CandidateOut
from src.plan.router import _parse_reason_detail, _summarize_cascade_candidates


def test_parse_reason_detail_converts_types():
    parsed = _parse_reason_detail(
        "chain_depth=2; assigned_steps=1; blocked_steps=1; uncovered_p4=0; disposed_p5=2; feasible_hard=false"
    )

    assert parsed["chain_depth"] == 2
    assert parsed["assigned_steps"] == 1
    assert parsed["blocked_steps"] == 1
    assert parsed["disposed_p5"] == 2
    assert parsed["feasible_hard"] is False


def test_summarize_cascade_candidates_aggregates_counts():
    c1 = CandidateOut(
        candidate_id="C1",
        driver_id="D1",
        est_cost=100.0,
        feasible_hard=True,
        reason_code="CASCADE_MULTI_DRIVER",
        reason_detail="chain_depth=2; assigned_steps=2; blocked_steps=0; uncovered_p4=0; disposed_p5=0; feasible_hard=true",
    )
    c2 = CandidateOut(
        candidate_id="C2",
        driver_id="D2",
        est_cost=120.0,
        feasible_hard=False,
        reason_code="CASCADE_PARTIAL_UNRESOLVED",
        reason_detail="chain_depth=1; assigned_steps=0; blocked_steps=1; uncovered_p4=1; disposed_p5=0; feasible_hard=false",
    )

    summary = _summarize_cascade_candidates([c1, c2])

    assert summary["candidates_total"] == 2
    assert summary["feasible_hard_count"] == 1
    assert summary["max_chain_depth"] == 2
    assert summary["uncovered_p4_total"] == 1
    assert summary["disposed_p5_total"] == 0
    assert summary["unresolved_total"] == 1
    assert summary["reason_code_counts"]["CASCADE_MULTI_DRIVER"] == 1
    assert summary["reason_code_counts"]["CASCADE_PARTIAL_UNRESOLVED"] == 1
