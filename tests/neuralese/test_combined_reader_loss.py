from contextlib import nullcontext

import torch

from natlang_neuralese.serve.grad import GradSession


def test_combined_reader_loss_keeps_objective_and_gradient_with_one_student_pass():
    torch.manual_seed(0)
    student = torch.randn(4, 7, requires_grad=True)
    teacher = torch.randn(4, 7)
    targets = torch.tensor([0, 1, 3, 5])
    calls = []

    class Session:
        def _adapted(self, *args):
            return nullcontext()

        def _target_items(self, messages, tools, target):
            return messages, target

        def _score(self, prompt, rest, leaves, write_terms=False):
            calls.append(prompt)
            logits = teacher if prompt == 'teacher' else student
            return {'token_logits': logits, 'token_logp': torch.log_softmax(logits, -1).gather(1, targets[:, None])[:, 0]}

    expected = torch.nn.functional.cross_entropy(student, targets)
    t, s = torch.log_softmax(teacher, -1), torch.log_softmax(student, -1)
    expected = expected + .7 * (t.exp() * (t-s)).sum(-1).mean()
    actual = GradSession.supervised_text_loss(Session(), {'messages': 'student', 'target': {}}, {},
                                              teacher_messages='teacher', distill_weight=.7)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, student, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(ga, ge)
    assert calls == ['teacher', 'student']
