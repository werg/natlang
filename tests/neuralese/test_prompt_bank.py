import torch

from natlang_neuralese.prompt_bank import PromptBank, find_pieces, load_bank, save_bank, soften
from natlang_neuralese.serve.store import content_id


def test_soften_matches_longest_pieces_and_leaves_replies_alone():
    texts = {"interpreter": "You run one call.", "interpreter/depth-limit": "You run one call. No more nl.", "notice": "[Last turn.]"}
    ids = {k: f"nz1_{k[:3]}" for k in texts}
    messages = [{"role": "system", "content": "You run one call. No more nl.\n\nGuidance."},
                {"role": "tool", "content": "ok[Last turn.]"},
                {"role": "assistant", "content": "You run one call."}]
    out, used = soften(messages, texts, ids)
    assert out[0]["content"] == [{"type": "neuralese", "id": ids["interpreter/depth-limit"]}, {"type": "text", "text": "\n\nGuidance."}]
    assert out[1]["content"][1] == {"type": "neuralese", "id": ids["notice"]}
    assert out[2] is messages[2]
    assert used == {"interpreter/depth-limit", "notice"}
    assert find_pieces("nothing", texts) == []


def test_saved_bank_round_trips_with_content_ids_and_init_texts(tmp_path):
    bank = PromptBank("nd:natlang@1", {"a": "alpha text", "b": "beta"}, {"a": torch.randn(3, 4), "b": torch.randn(2, 4)})
    trained = torch.randn(3, 4)
    ids = save_bank(tmp_path / "bank.nz", bank, {"a": trained}, {"kind": "test"})
    loaded = load_bank(tmp_path / "bank.nz")
    assert loaded.texts == bank.texts and loaded.dialect == bank.dialect
    assert torch.equal(loaded.rows["a"], trained) and torch.equal(loaded.rows["b"], bank.rows["b"])
    assert ids["a"] == content_id("nd:natlang@1", 3, 4, "f32", trained.numpy().astype("<f4").tobytes())
    assert loaded.ids == ids
