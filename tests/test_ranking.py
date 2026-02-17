from app.models.schemas import Candidate
from app.services.ranking import weighted_reciprocal_rank_fusion


def test_weighted_rrf_basic():
    elastic = [Candidate(id="1", source="elastic"), Candidate(id="2", source="elastic")]
    vector = [Candidate(id="2", source="vector"), Candidate(id="3", source="vector")]
    graph = [Candidate(id="3", source="graph"), Candidate(id="1", source="graph")]

    fused = weighted_reciprocal_rank_fusion(
        {"elastic": elastic, "vector": vector, "graph": graph},
        {"elastic": 0.3, "vector": 0.5, "graph": 0.2},
    )
    assert len(fused) == 3
    assert fused[0].id in {"2", "3", "1"}
