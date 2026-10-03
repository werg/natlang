"""CPU regression for append order and checkpoint continuity across two appends.

This exercises the transition math and serialized optimizer/RNG state without
loading a student model. The CLI currently accepts only one append transition
per lineage; this test documents the intended two-boundary continuation math,
not support for chained production append manifests.
"""
import copy
import unittest

import torch

from scripts.training_append import (
    append_train_order,
    cosine_extension_multiplier,
    ids_digest,
)


class AppendResumeTests(unittest.TestCase):
    def test_two_appends_preserve_order_optimizer_rng_cursor_and_lr_boundaries(self):
        torch.set_num_threads(1)
        base = [{"id": f"b{i}", "value": float(i + 1)} for i in range(4)]
        add1 = [{"id": f"a{i}", "value": float(i + 5)} for i in range(2)]
        add2 = [{"id": f"z{i}", "value": float(i + 7)} for i in range(2)]
        order1 = append_train_order(
            base, add1, base + add1,
            base_order_sha256=ids_digest([r["id"] for r in base]),
            candidate_order_sha256=ids_digest([r["id"] for r in add1]),
        )
        order2 = append_train_order(
            order1, add2, base + add1 + add2,
            base_order_sha256=ids_digest([r["id"] for r in order1]),
            candidate_order_sha256=ids_digest([r["id"] for r in add2]),
        )
        self.assertEqual([r["id"] for r in order2],
                         ["b0", "b1", "b2", "b3", "a0", "a1", "z0", "z1"])

        torch.manual_seed(719)
        initial = torch.nn.Linear(1, 1, bias=True)
        initial_state = copy.deepcopy(initial.state_dict())
        start_rng = torch.get_rng_state().clone()

        # The same piecewise cosine extension policy is used by both paths:
        # append one anchors at step 2 and append two re-anchors at step 4.
        def multiplier(step):
            if step < 2:
                return 1.0
            if step < 4:
                return cosine_extension_multiplier(step, start_step=2, end_step=6)
            return cosine_extension_multiplier(step, start_step=4, end_step=8) * 0.5

        def run(resume_boundaries):
            model = torch.nn.Linear(1, 1, bias=True)
            model.load_state_dict(copy.deepcopy(initial_state))
            torch.set_rng_state(start_rng)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
            cursor = 0
            seen_lrs = []
            for step in range(8):
                if step in resume_boundaries:
                    checkpoint = {
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "rng": torch.get_rng_state().clone(),
                        "cursor": cursor,
                        "step": step,
                    }
                    # Rebuild exactly as a resumed process would, then restore
                    # optimizer slots, random stream, and sample cursor.
                    resumed = torch.nn.Linear(1, 1, bias=True)
                    resumed.load_state_dict(checkpoint["model"])
                    model = resumed
                    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    torch.set_rng_state(checkpoint["rng"])
                    cursor = checkpoint["cursor"]
                    self.assertEqual(cursor, step)
                lr = 0.03 * multiplier(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                seen_lrs.append(lr)
                value = order2[cursor]["value"]
                noise = torch.rand(())
                x = torch.tensor([[value / 10.0]])
                target = torch.tensor([[0.25 + noise.item() / 10.0]])
                loss = ((model(x) - target) ** 2).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                cursor += 1
            return model, optimizer, torch.get_rng_state().clone(), cursor, seen_lrs

        uninterrupted = run(set())
        appended = run({2, 4})
        self.assertEqual(uninterrupted[3], 8)
        self.assertEqual(appended[3], 8)
        self.assertEqual(uninterrupted[4], appended[4])
        self.assertAlmostEqual(appended[4][2], 0.03, places=12)
        self.assertAlmostEqual(appended[4][4], 0.015, places=12)
        self.assertTrue(torch.equal(uninterrupted[2], appended[2]))
        for key, value in uninterrupted[0].state_dict().items():
            self.assertTrue(torch.equal(value, appended[0].state_dict()[key]), key)
        # Optimizer moments and step counters must survive both reconstruction
        # boundaries exactly.
        left, right = uninterrupted[1].state_dict(), appended[1].state_dict()
        self.assertEqual(left["param_groups"], right["param_groups"])
        self.assertEqual(left["state"].keys(), right["state"].keys())
        for pid in left["state"]:
            for key, value in left["state"][pid].items():
                other = right["state"][pid][key]
                self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)


if __name__ == "__main__":
    unittest.main()
