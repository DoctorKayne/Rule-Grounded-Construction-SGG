import torch
import torch.nn.functional as F

from rule_grounded_sgg.assignment import PPEAssignment


def test_assignment_is_softmax_over_workers_for_each_ppe():
    module = PPEAssignment(hidden_dim=8, evidence_dim=8, dropout=0.0)
    object_states = torch.randn(3, 8, requires_grad=True)
    pair_index = torch.tensor([[0, 2], [1, 2]], dtype=torch.long)
    evidence = torch.randn(2, 8)
    result = module(
        object_states,
        pair_index,
        evidence,
        [{"ppe": 2, "workers": [0, 1], "target_position": 0}],
    )[0]
    assert result["probabilities"].shape == (2,)
    assert torch.allclose(result["probabilities"].sum(), torch.tensor(1.0), atol=1e-6)

    loss = F.cross_entropy(
        result["logits"].unsqueeze(0), torch.tensor([result["target_position"]])
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in module.parameters())
    assert object_states.grad is not None
    assert torch.isfinite(object_states.grad).all()
