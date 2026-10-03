"""CPU proof of the real Muon optimizer and append scheduler resume path."""
import copy
import unittest

import torch

from scripts.training_append import cosine_extension_multiplier
from scripts.training_optimizers import make_muon_optimizer


@unittest.skipUnless(hasattr(torch.optim, "Muon"), "requires the training image's native Muon")
class MuonAppendResumeTests(unittest.TestCase):
    def test_append_anchor_and_resume_preserve_muon_and_auxiliary_state(self):
        torch.set_num_threads(1)
        torch.manual_seed(761)
        initial = copy.deepcopy(torch.nn.Linear(4, 3).state_dict())
        initial_rng = torch.get_rng_state().clone()

        def run(reconstruct):
            model = torch.nn.Linear(4, 3)
            model.load_state_dict(initial)
            torch.set_rng_state(initial_rng)
            opt = make_muon_optimizer(model, lr=0.003)
            extension = None

            def multiplier(step):
                if extension is not None:
                    return cosine_extension_multiplier(step, start_step=3, end_step=9)
                return 1 - step / 10

            sched = torch.optim.lr_scheduler.LambdaLR(opt, multiplier)
            rates = []
            for step in range(9):
                if step == 3:
                    anchor = [group["lr"] for group in opt.param_groups]
                    extension = {"start_step": 3, "end_step": 9, "start_lrs": anchor}
                    sched.base_lrs = anchor
                    for group, lr in zip(opt.param_groups, anchor):
                        group["initial_lr"] = lr
                if reconstruct and step in (3, 6):
                    saved_model = copy.deepcopy(model.state_dict())
                    saved_opt = copy.deepcopy(opt.state_dict())
                    saved_sched = copy.deepcopy(sched.state_dict())
                    saved_rng = torch.get_rng_state().clone()
                    model = torch.nn.Linear(4, 3)
                    model.load_state_dict(saved_model)
                    opt = make_muon_optimizer(model, lr=0.003)
                    sched = torch.optim.lr_scheduler.LambdaLR(opt, multiplier)
                    opt.load_state_dict(saved_opt)
                    sched.load_state_dict(saved_sched)
                    torch.set_rng_state(saved_rng)
                    sched.base_lrs = extension["start_lrs"]
                    self.assertEqual(sched.last_epoch, step)
                rates.append([g["lr"] for g in opt.param_groups])
                x = torch.rand(2, 4)
                loss = model(x).square().mean()
                loss.backward()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            return model.state_dict(), opt.state_dict(), sched.state_dict(), rates, torch.get_rng_state()

        expected, actual = run(False), run(True)
        self.assertEqual(expected[3], actual[3])
        self.assertTrue(torch.equal(expected[4], actual[4]))
        for name, value in expected[0].items():
            self.assertTrue(torch.equal(value, actual[0][name]), name)
        for child in ("muon", "adamw"):
            self.assertEqual(expected[1][child]["param_groups"], actual[1][child]["param_groups"])
            for pid, slots in expected[1][child]["state"].items():
                for name, value in slots.items():
                    other = actual[1][child]["state"][pid][name]
                    self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)
        self.assertEqual(expected[2], actual[2])


if __name__ == "__main__":
    unittest.main()
