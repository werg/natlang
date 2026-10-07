"""Gradient replay sessions and optimiser steps (S4 §4.4; spec/NEURALESE_GRAPH.md, "Replay").

A `grad` request replays recorded model turns: each term names a recorded prompt (messages with block parts, whose
tool results are the recorded effect results) and the assistant output to score (the recorded output, or an expected
one). Discrete choices are held fixed: the output's text tokens are teacher-forced, and every block it wrote is
re-written with its recorded length (the stop decisions are scored, not re-sampled) and then read back with its
recorded payload. The differentiable path is recomputed with the requested **argument** blocks as leaf tensors;
gradients are returned as new store entries in the `#grad` variant of the dialect, so they cannot be read as values.

Terms (`kind`):

| Kind | Loss |
| --- | --- |
| `crossEntropy` | Mean negative log-probability of the target's text tokens. |
| `logLikelihood` | Negative log-probability of the whole recorded output: text tokens, each written block's stop decisions, and, for blocks written at Neuralese temperature > 0, the Gaussian log-density of the recorded payload under the recomputed mean and scale (`heads.payload_log_prob`). |
| `selfDistill` | KL(teacher ‖ student) over the target's text positions; the teacher is the same model given `teacher_messages` (the full source), without gradient. The reference server has no separately trained deltas, so "deltas off" is the same weights. |
| `klPrior` | KL(N(μ, σ²) ‖ N(0, I)) of Gaussian blocks `{mean, log_sigma}` in normalised space (`heads.payload_kl`). |
| `decision` | A proper scoring rule on the decision readout: `options` (assistant replies) are scored after `messages` as in `/v1/neuralese/decide`, normalised over the options, and compared with `target` probabilities, or with the readout of a `teacher` (`{messages, adapters}`: the same model given privileged context, scored without gradient; conditioned distillation) (`rule`: `logLoss`, the default, is cross-entropy against the target distribution; `brier` is the squared error; `rps` is the ranked probability score for ordered options; `expectedReward` takes `target.rewards`, one per option, and its loss is −Σ pᵢ rᵢ, the exact policy objective of a finite decision). |

Adapters (`model/tiny_adapters.py`): a request's `adapters` (`[{"id", "scale"}]`, or a term's own `adapters`) are
active in every forward of its terms, so a term scores the adapted model. Adapter blocks may be `arguments`: their
coefficients are then leaves and get gradients like any other block (in the adapter's own `#grad` dialect). A
`selfDistill` teacher runs without adapters unless the term names `teacher_adapters`: the teacher is the base model
given the privileged context.

Every term has a `weight`; the session loss is the weighted sum. Constants (`stopGradient`, inner gradients of a
first-order nested `grad`) are simply blocks that are not arguments.

Exact second order (`order: 2`, with `derived`): the inner computations the loss ran on its way, in order, recomputed
differentiably before the terms: `{"kind": "grad", "arguments", "terms", "producers", "gradients": {argument: gradient
block}}` (an inner gradient session; its gradients are taken with `create_graph`, so they are functions of whatever
the inner arguments and terms depend on) and `{"kind": "optim", "optimizer", "hyper", "params", "grads", "state",
"results": {"params": [...], "state": {...}}}` (an optimiser step, applied functionally to those tensors). Each derived
block that depends on an argument stands for its recorded block in everything after it, so the outer terms
differentiate through the inner updates (meta-learning through update steps, S6 §5.5). The recorded values are kept
(as for re-written producers), so the loss is the recorded one. Attention runs in its math kernel and the
convolution without the fused kernel in these sessions (double backward).

Producers (`producers`: recorded turns `{messages, tools, reply, adapters}` that wrote blocks; spec/NEURALESE_GRAPH.md,
"Replay", step 4): a block a term reads (in its prompt, or written earlier in its target) that one of these turns
wrote is re-written from that turn, differentiably, when the turn's context depends on an argument (directly, through
an argument adapter, or through another re-written block): its prompt and the reply up to the block are prefilled
with gradient and the write unrolled at the recorded length. The block keeps its recorded value (`recorded +
(rewritten − rewritten.detach())`), so replay changes no observation, and the gradient flows through the writer into
the producing turn's context. This is how a call that returns a Neuralese value (a template readout) is trained by
its caller's loss: calling a function, retrieving its value and splicing it into the caller's trajectory.

`optim_step` applies SGD (with momentum) or Adam to argument blocks with their gradients and returns new parameter
blocks and new optimiser-state blocks: nothing is updated in place.
"""

from __future__ import annotations

import contextlib
import math

import torch

from ..model.heads import PayloadSample, payload_kl, payload_log_prob
from ..model.lfm2_port import PortCache
from .chat import RequestError, opens_thinking, render_messages, render_with_empty_thought
from .store import Block, make_block


class Unavailable(RequestError):
    def __init__(self, detail: str):
        super().__init__("neuralese-grad-unavailable", detail)


def grad_dialect(dialect: str) -> str:
    return f"{dialect}#grad"


def state_dialect(dialect: str) -> str:
    return f"{dialect}#opt"


class GradSession:
    """Replays recorded turns on one engine's backbone and heads."""

    def __init__(self, engine):
        self.engine = engine
        self.backbone, self.heads = engine.backbone, engine.heads

    # Sequences -----------------------------------------------------------------------------
    def _items(self, segments, blocks: list[str], escape_nonce: str = "") -> list:
        items = []
        for segment in segments:
            if isinstance(segment, str):
                items.extend(("tok", i) for i in self.engine._template_tokens(segment, escape_nonce))
            else:
                items.append(("block", blocks[segment]))
        return items

    def _target_items(self, messages, tools, target) -> tuple[list, list]:
        """(prompt items, target items) with the target teacher-forced after the generation prompt."""
        engine = self.engine
        prompt = render_messages(messages, tools, engine._template, engine.specials, block_type=engine.block_value_type)

        thinking = opens_thinking(engine._template([{"role": "user", "content": "x"}], None))

        def full_template(msgs, tls):
            render = lambda m: engine.tokenizer.apply_chat_template(m, tools=tls or None, tokenize=False,
                                                                     add_generation_prompt=False)
            return render_with_empty_thought(render, msgs) if thinking else render(msgs)

        full = render_messages(list(messages) + [target], tools, full_template, engine.specials, block_type=engine.block_value_type)
        before = self._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
        after = self._items(full.segments, full.blocks, full.escape_nonce)
        # The prompt is scored exactly as inference rendered it; the template may render earlier turns differently
        # once another assistant turn follows (it drops past reasoning), so the target is cut from the full rendering
        # after its own generation prefix rather than by matching the whole prompt.
        im_start = engine.tokenizer.convert_tokens_to_ids("<|im_start|>")
        starts = [i for i, item in enumerate(before) if item == ("tok", im_start)]
        if not starts:
            raise RequestError("neuralese-grad-target", "the rendered prompt has no generation prefix")
        prefix = before[starts[-1]:]
        found = [i for i in range(len(after) - len(prefix), -1, -1) if after[i:i + len(prefix)] == prefix]
        if found:
            rest = after[found[0] + len(prefix):]
        else:
            rest = self._split_boundary(prefix, after)
            if rest is None:
                raise RequestError("neuralese-grad-target", "the target's rendering has no assistant prefix")
        im_end = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ends = [i for i, item in enumerate(rest) if item == ("tok", im_end)]
        if ends:
            rest = rest[:ends[-1] + 1]
        return before, rest

    def _split_boundary(self, prefix: list, after: list) -> list | None:
        """The target after the generation prefix when tokenization merged the prefix's last token with the reply's
        first (a thinking prompt ends `<think>\\n` and the reply starts `\\n</think>`, which the full rendering
        tokenizes as one `\\n\\n`). Generation continues from the prompt's own tokens, so the merged token's remainder
        is retokenized on its own: exactly what the model is scored on after the prompt."""
        head, last = prefix[:-1], prefix[-1]
        if last[0] != "tok":
            return None
        decode = lambda token: self.engine.tokenizer.decode([token])
        tail = decode(last[1])
        for i in range(len(after) - len(prefix), -1, -1):
            joined = after[i + len(head)] if i + len(head) < len(after) else None
            if after[i:i + len(head)] == head and joined is not None and joined[0] == "tok":
                text = decode(joined[1])
                if text.startswith(tail) and len(text) > len(tail):
                    return [("tok", t) for t in self.engine._tokens(text[len(tail):])] + after[i + len(head) + 1:]
        return None

    def _payload(self, block_id: str, leaves: dict) -> torch.Tensor:
        if block_id in leaves:
            return leaves[block_id]
        return self.engine.lookup(block_id).payload.clone().to(self.engine.device)  # stored rows may be inference tensors

    def _embed_items(self, items, leaves, starts: list | None = None) -> torch.Tensor:
        """Input embeddings of the items; `starts`, when given, receives each item's first position."""
        backbone, heads, device = self.backbone, self.heads, self.engine.device
        dtype = backbone.embedding_weight.dtype
        pieces, run = [], []
        position = 0

        def flush():
            if run:
                pieces.append(backbone.embed(torch.tensor([run], device=device)))
                run.clear()

        for kind, value in items:
            if starts is not None:
                starts.append(position)
            if kind == "tok":
                run.append(value)
                position += 1
            else:
                flush()
                pieces.append(heads.read_embeddings(backbone, self._payload(value, leaves).to(dtype)[None]))
                position += pieces[-1].shape[1]
        flush()
        return torch.cat(pieces, 1)

    def _context_weights(self, prompt, feedback_weight: float = 1.0) -> list[float]:
        """Per-item CE weights of the prompt's text. Instructions and inputs (everything before the first assistant
        reply) are NatLang programs the student should learn to write: weight 1. Later non-assistant turns are
        mechanical feedback (tool results, harness nudges): `feedback_weight`. Items up to the last earlier assistant
        reply weigh 0: their own records already supervise them. The generation prefix weighs 1."""
        tokenizer = self.engine.tokenizer
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        starts = [i for i, item in enumerate(prompt) if item == ("tok", im_start)]
        role = lambda i: tokenizer.decode([v for k, v in prompt[i + 1:i + 3] if k == "tok"]).lstrip()
        assistants = [i for i in starts[:-1] if role(i).startswith("assistant")]
        weights = [1.0] * len(prompt)
        if not assistants:
            return weights
        bounds = starts + [len(prompt)]
        for begin, end in zip(bounds, bounds[1:]):
            if begin == starts[-1]:
                continue  # the generation prefix
            weight = 0.0 if begin <= assistants[-1] else 1.0 if begin < assistants[0] else feedback_weight
            weights[begin:end] = [weight] * (end - begin)
        weights[:starts[0]] = [0.0] * starts[0]
        return weights

    def _context_logp(self, h, prompt, starts, weights, chunk: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        """(log-probabilities, weights) of the prompt's positively weighted text tokens, read from the prompt pass's
        final states in checkpointed chunks (no prompt-length x vocabulary activation is retained). Block payloads
        are not tokens and are never targets; the token after a block is predicted from the block's close."""
        index = [j for j in range(1, len(prompt)) if prompt[j][0] == "tok" and weights[j] > 0]
        if not index:
            return h.new_zeros(0, dtype=torch.float32), h.new_zeros(0, dtype=torch.float32)
        device = h.device
        positions = torch.tensor([starts[j] - 1 for j in index], device=device)
        targets = torch.tensor([prompt[j][1] for j in index], device=device)
        logits_of = self.backbone.logits

        def logp(states, gold):
            return torch.log_softmax(logits_of(states[None]).float()[0], -1).gather(1, gold[:, None])[:, 0]

        from torch.utils.checkpoint import checkpoint
        out = []
        for begin in range(0, len(index), chunk):
            states = h[0, positions[begin:begin + chunk]]
            gold = targets[begin:begin + chunk]
            out.append(checkpoint(logp, states, gold, use_reentrant=False) if torch.is_grad_enabled()
                       else logp(states, gold))
        return torch.cat(out), torch.tensor([weights[j] for j in index], device=device, dtype=torch.float32)

    def _score(self, prompt, target, leaves, write_terms: bool, collect_token_states: bool = False,
               context_weights: list[float] | None = None) -> dict:
        """Teacher-forced pass over prompt + target. Returns per-position text log-probs (and logits) for the
        target's text tokens, plus write terms for written blocks; with per-prompt-item `context_weights`, also
        `context_logp` and `context_weight` for the prompt's positively weighted text tokens."""
        backbone, heads = self.backbone, self.heads
        starts = [] if context_weights is not None else None
        embeds = self._embed_items(prompt, leaves, starts)
        # Only the last prompt position's logits are needed: a long prompt's full vocabulary projection is large.
        out = backbone.forward_embeds(embeds, logits=False, cutoff=heads.cutoff if not heads.read_markers else None)
        context_logp, context_weight = (self._context_logp(out["h_final"], prompt, starts, context_weights)
                                        if context_weights is not None else (None, None))
        cache, last = out["cache"], backbone.logits(out["h_final"][:, -1:])[:, -1]
        cut_state = out['h_cut'][:, -1] if not heads.read_markers else None
        top_state = out['h_final'][:, -1] if not heads.read_markers else None
        token_logp, token_logits, write_logp, stop_states = [], [], [], []
        index = 0
        while index < len(target):
            kind, value = target[index]
            if kind == "tok":
                run = []
                while index < len(target) and target[index][0] == "tok":
                    run.append(target[index][1])
                    index += 1
                ids = torch.tensor([run], device=self.engine.device)
                step = backbone.forward_ids(ids, cache=cache, cutoff=heads.cutoff if not heads.read_markers else None)
                logits = torch.cat([last[:, None], step["logits"][:, :-1]], 1)[0]
                if collect_token_states:
                    stop_states.append(heads.stop_states(step['h_cut'], step['h_final']))
                token_logits.append(logits)
                token_logp.append(torch.log_softmax(logits.float(), -1).gather(1, ids[0][:, None])[:, 0])
                cache, last = step["cache"], step["logits"][:, -1]
                if not heads.read_markers:
                    cut_state, top_state = step['h_cut'][:, -1], step['h_final'][:, -1]
                continue
            # A written block: the open decision is a text decision; the write is replayed with its recorded length.
            if heads.read_markers:
                open_id = backbone.controls.open_id
                token_logits.append(last)
                token_logp.append(torch.log_softmax(last.float(), -1)[:, open_id])
                opened = backbone.forward_ids(torch.tensor([[open_id]], device=self.engine.device), cache=cache,
                                              cutoff=heads.cutoff)
                block_start, state, top = opened["cache"], opened["h_cut"][:, -1], None
            else:
                block_start, state, top = cache, cut_state, top_state
            if write_terms:
                # Replaying the recorded write needs the stored block; a leaf (a value written afresh) does not.
                write_logp.append(self._replay_write(self.engine.lookup(value), block_start, state, top))
            payload = self._payload(value, leaves)[None].to(backbone.embedding_weight.dtype)
            back = backbone.forward_embeds(heads.read_embeddings(backbone, payload, close_only=True),
                                           cache=block_start, cutoff=heads.cutoff if not heads.read_markers else None)
            cache, last = back["cache"], back["logits"][:, -1]
            if not heads.read_markers:
                cut_state, top_state = back['h_cut'][:, -1], back['h_final'][:, -1]
            index += 1
        return {"token_logp": torch.cat(token_logp) if token_logp else torch.zeros(0),
                "token_logits": torch.cat(token_logits) if token_logits else None,
                "write_logp": torch.stack(write_logp).sum() if write_logp else torch.zeros(()),
                "token_stop_states": torch.cat(stop_states, 1) if stop_states else None,
                "context_logp": context_logp, "context_weight": context_weight}

    def _replay_write(self, block: Block, block_start: PortCache, state: torch.Tensor,
                      top: torch.Tensor | None = None) -> torch.Tensor:
        """Log-probability of a recorded write: stop decisions at the recorded length, and the payload density when
        it was sampled at temperature > 0."""
        backbone, heads = self.backbone, self.heads
        producer = block.producer or {}
        length = block.length
        cache, sketches, shallow, logp = block_start, [], [], []
        final_source = heads.stop_source == "final"
        for count in range(length + 1):
            if count > 0 and not final_source:
                logit = heads.stop(state, torch.full((1,), count, device=state.device, dtype=torch.long))[0].float()
                stopped = count == length and not block.truncated
                logp.append(torch.nn.functional.logsigmoid(logit if stopped else -logit))
            if count == length:
                break
            sketch = heads.feedback(state)
            h, cache = backbone.run_layers(sketch[:, None], range(0, heads.cutoff), cache)
            sketches.append(sketch)
            shallow.append(h[:, 0])
            state = h[:, 0]
        if final_source and length:
            # Decisions after c vectors read the completed state h_D[c - 1]: continue before the length, stop at it.
            final, _ = backbone.run_layers(torch.stack(shallow, 1), range(heads.cutoff, backbone.num_layers), block_start)
            counts = torch.arange(1, length + 1, device=state.device)[None]
            logits = heads.stop(final, counts)[0].float()
            for c in range(1, length + 1):
                if c == length and block.truncated:
                    break
                stopped = c == length
                logp.append(torch.nn.functional.logsigmoid(logits[c - 1] if stopped else -logits[c - 1]))
        total = torch.stack(logp).sum() if logp else torch.zeros((), device=state.device)
        tau = float(producer.get("temperature") or 0.0)
        if tau > 0 and length:
            final, _ = backbone.run_layers(torch.stack(shallow, 1), range(heads.cutoff, backbone.num_layers),
                                           block_start)
            mu, log_sigma = heads.content.distribution(torch.stack(sketches, 1), heads.payload_states(final, top))
            sample = PayloadSample(mu, mu, log_sigma, None, tau)
            recorded = block.payload.clone().to(mu.device, mu.dtype)[None]
            total = total + payload_log_prob(sample, recorded).sum()
        return total

    # Decision readout ----------------------------------------------------------------------
    def decision_prepare(self, messages, tools, options: list[str], leaves: dict):
        """Shared state of a decision: the prompt run once (and the tokens every option shares at the start of the
        reply), and each option's own tokens. Returns (cache, last logits, own token lists)."""
        if not options:
            raise RequestError("neuralese-decision", "a decision needs at least one option")
        prompt, rests = None, []
        for option in options:
            if not isinstance(option, str):
                raise RequestError("neuralese-decision", "options are reply texts")
            before, rest = self._target_items(messages, tools, {"role": "assistant", "content": option})
            if prompt is not None and before != prompt:
                raise RequestError("neuralese-decision", "options rendered different prompts")
            if any(kind != "tok" for kind, _ in rest):
                raise RequestError("neuralese-decision", "options must be text")
            prompt = before
            rests.append([value for _, value in rest])
        shared = 0
        if len(rests) > 1:
            while all(len(r) > shared for r in rests) and all(r[shared] == rests[0][shared] for r in rests):
                shared += 1
        out = self.backbone.forward_embeds(self._embed_items(prompt, leaves), logits=False)
        cache, last = out["cache"], self.backbone.logits(out["h_final"][:, -1:])[:, -1]
        if shared:
            step = self.backbone.forward_ids(torch.tensor([rests[0][:shared]], device=self.engine.device), cache=cache)
            cache, last = step["cache"], step["logits"][:, -1]
        return cache, last, [r[shared:] for r in rests]

    def option_logprob(self, cache, last, own: list[int]) -> torch.Tensor:
        """Log-probability of one option's own tokens, continued from the shared cache."""
        if not own:
            return torch.zeros((), device=last.device)
        ids = torch.tensor([own], device=self.engine.device)
        step = self.backbone.forward_ids(ids, cache=cache)
        logits = torch.cat([last[:, None], step["logits"][:, :-1]], 1)[0]
        return torch.log_softmax(logits.float(), -1).gather(1, ids[0][:, None]).sum()

    def decision_logprobs(self, messages, tools, options: list[str], leaves: dict) -> tuple[torch.Tensor, list[int]]:
        """Total log-probability of each option as the whole assistant reply. The prompt is run once; each option
        continues from its cache. Tokens all options share at the start of the reply are left out."""
        cache, last, owns = self.decision_prepare(messages, tools, options, leaves)
        return torch.stack([self.option_logprob(cache, last, own) for own in owns]), [len(own) for own in owns]

    def decision_backward(self, messages, tools, options: list[str], goal: torch.Tensor, scale: float = 1.0,
                          leaves: dict | None = None) -> float:
        """Backpropagate the cross-entropy of the normalised option distribution against `goal`, holding one option's
        graph at a time: scores are computed without gradient, then each option is re-run and backpropagated with
        its coefficient softmax_i - goal_i (the gradient of the loss in its score). Gradients reach `leaves` (block ID →
        tensor) like any other parameter. Returns the loss."""
        leaves = leaves or {}
        with torch.no_grad():
            scores, _ = self.decision_logprobs(messages, tools, options, leaves)
        goal = goal / goal.sum()
        logp = torch.log_softmax(scores, 0)
        coefficients = (logp.exp() - goal) * scale
        cache, last, owns = self.decision_prepare(messages, tools, options, leaves)
        for index, own in enumerate(owns):
            if float(coefficients[index]) == 0.0:
                continue
            (coefficients[index] * self.option_logprob(cache, last, own)).backward(retain_graph=True)
        # The prompt's graph is shared by every option; release it after the last option.
        del cache, last
        return float(-(goal * logp).sum())

    def _decision_term(self, term: dict, leaves: dict) -> torch.Tensor:
        options, target, teacher = term.get("options") or [], term.get("target"), term.get("teacher")
        rule = term.get("rule") or "logLoss"
        if rule == "expectedReward":
            # Reinforcement on a finite decision: the expected reward under the readout is exact, so its gradient is
            # the policy gradient without sampling. Loss = −Σ p_i r_i.
            rewards = target.get("rewards") if isinstance(target, dict) else None
            if not isinstance(rewards, list) or len(rewards) != len(options):
                raise RequestError("neuralese-grad-term", "expectedReward needs target.rewards, one per option")
            scores, _ = self.decision_logprobs(term.get("messages") or [], term.get("tools"), options, leaves)
            p = torch.softmax(scores.float(), 0)
            return -(p * torch.tensor([float(r) for r in rewards], device=p.device)).sum()
        probabilities = target.get("probabilities") if isinstance(target, dict) else None
        if probabilities is None and isinstance(teacher, dict):
            # Conditioned distillation: the target is the same model's readout given the teacher's messages (its
            # privileged context), with the teacher's adapters (none by default), without gradient.
            with torch.no_grad(), self._adapted(teacher.get("adapters"), {}):
                teacher_scores, _ = self.decision_logprobs(teacher.get("messages") or [], teacher.get("tools"), options, {})
            probabilities = torch.softmax(teacher_scores.float(), 0).tolist()
        if not isinstance(probabilities, list) or len(probabilities) != len(options):
            raise RequestError("neuralese-grad-term", "decision needs target.probabilities (one per option) or a teacher")
        scores, _ = self.decision_logprobs(term.get("messages") or [], term.get("tools"), options, leaves)
        logp = torch.log_softmax(scores, 0)
        goal = torch.tensor([float(p) for p in probabilities], device=logp.device)
        goal = goal / goal.sum()
        if rule == "logLoss":
            return -(goal * logp).sum()
        if rule == "brier":
            return ((logp.exp() - goal) ** 2).sum()
        if rule == "rps":
            return ((torch.cumsum(logp.exp(), 0) - torch.cumsum(goal, 0))[:-1] ** 2).sum() / max(1, len(options) - 1)
        raise RequestError("neuralese-grad-term", f"unknown decision rule {rule!r}")

    # Terms ---------------------------------------------------------------------------------
    def _term(self, term: dict, leaves: dict) -> torch.Tensor:
        kind = term.get("kind")
        if kind == "klPrior":
            losses = []
            for entry in term.get("blocks") or []:
                mean = self._payload(entry["mean"], leaves).float()
                log_sigma = (self._payload(entry["log_sigma"], leaves).float() if entry.get("log_sigma")
                             else torch.zeros_like(mean))
                losses.append(payload_kl(PayloadSample(mean, mean, log_sigma, None, 1.0)).mean())
            return torch.stack(losses).mean() if losses else torch.zeros(())
        if kind == "decision":
            return self._decision_term(term, leaves)
        messages, tools, target = term.get("messages") or [], term.get("tools"), term.get("target")
        if not isinstance(target, dict):
            raise RequestError("neuralese-grad-term", f"{kind} needs a target assistant message")
        prompt, rest = self._target_items(messages, tools, target)
        if kind in ("crossEntropy", "logLikelihood"):
            scored = self._score(prompt, rest, leaves, write_terms=kind == "logLikelihood")
            if kind == "crossEntropy":
                return -scored["token_logp"].mean()
            return -(scored["token_logp"].sum() + scored["write_logp"])
        if kind == "selfDistill":
            teacher_messages = term.get("teacher_messages")
            if not teacher_messages:
                raise RequestError("neuralese-grad-term", "selfDistill needs teacher_messages")
            student = self._score(prompt, rest, leaves, write_terms=False)["token_logits"]
            with torch.no_grad(), self._adapted(term.get("teacher_adapters"), {}):
                t_prompt, t_rest = self._target_items(teacher_messages, tools, target)
                teacher = self._score(t_prompt, t_rest, {}, write_terms=False)["token_logits"]
            if student is None or teacher is None or student.shape != teacher.shape:
                raise RequestError("neuralese-grad-term", "student and teacher targets do not align")
            t = torch.log_softmax(teacher.float(), -1)
            s = torch.log_softmax(student.float(), -1)
            return (t.exp() * (t - s)).sum(-1).mean()
        raise RequestError("neuralese-grad-term", f"unknown term kind {kind!r}")

    def supervised_continuation_loss(self, messages, tools, prefix, continuation, leaves, *, text_weight=1., stop_weight=0.):
        """Score only continuation tokens under the exact forced generation prefix.

        Tokenize prefix and value separately, as the writer does. Whole native
        replies can merge their boundary token and supervise a different state.
        """
        prompt = render_messages(messages, tools, self.engine._template, self.engine.specials,
                                 block_type=self.engine.block_value_type)
        before = self._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
        before += [("tok", t) for t in self.engine._tokens(prefix)]
        target = [("tok", t) for t in self.engine._tokens(continuation)]
        if not target:
            raise RequestError('neuralese-grad-term', 'forced continuation supervision needs target tokens')
        if stop_weight and self.heads.read_markers:
            raise RequestError('neuralese-grad-term', 'gold native stop supervision requires raw-token transport')
        scored = self._score(before, target, leaves, write_terms=False, collect_token_states=bool(stop_weight))
        loss = -text_weight * scored['token_logp'].mean()
        if stop_weight:
            states = scored['token_stop_states']
            counts = torch.arange(1, len(target) + 1, device=states.device)[None].expand(states.shape[0], -1)
            logits = self.heads.stop(states, counts).float()
            # Balance the one terminal event against the mean continuation event,
            # rather than reducing its influence by the gold body's length.
            terminal = torch.nn.functional.softplus(-logits[:, -1]).mean()
            boundary = terminal
            if len(target) > 1:
                boundary = (terminal + torch.nn.functional.softplus(logits[:, :-1]).mean()) / 2
            loss = loss + stop_weight * boundary
        return loss

    def supervised_text_loss(self, term, leaves, *, teacher_messages=None, distill_weight=0.0, context_weight=0.0,
                             feedback_weight=1.0):
        """CE and optional KL from one reader forward, with the same existing objectives.

        `context_weight` adds the CE of the prompt's new text (instructions and inputs, plus tool results and other
        mechanical feedback at `feedback_weight`), averaged over its tokens: the whole trajectory is a training
        target, not only the reply.

        Trajectory training used to replay the entire student reader once for CE
        and again for self-distillation, retaining both recurrence graphs.
        """
        messages, tools, target = term.get('messages') or [], term.get('tools'), term.get('target')
        teacher = None
        if distill_weight:
            if not teacher_messages:
                raise RequestError('neuralese-grad-term', 'distillation needs teacher_messages')
            with torch.no_grad(), self._adapted(term.get('teacher_adapters'), {}):
                tp, tr = self._target_items(teacher_messages, tools, target)
                teacher = self._score(tp, tr, {}, write_terms=False)['token_logits']
        prompt, rest = self._target_items(messages, tools, target)
        if context_weight:
            scored = self._score(prompt, rest, leaves, write_terms=False,
                                 context_weights=self._context_weights(prompt, feedback_weight))
        else:
            scored = self._score(prompt, rest, leaves, write_terms=False)
        loss = -scored['token_logp'].mean()
        if context_weight and scored['context_logp'].numel():
            loss = loss - context_weight * (scored['context_weight'] * scored['context_logp']).mean()
        if distill_weight:
            student = scored['token_logits']
            if student is None or teacher is None or student.shape != teacher.shape:
                raise RequestError('neuralese-grad-term', 'student and teacher targets do not align')
            t = torch.log_softmax(teacher.float(), -1)
            s = torch.log_softmax(student.float(), -1)
            loss = loss + distill_weight * (t.exp() * (t - s)).sum(-1).mean()
        return loss

    def _adapted(self, adapters, leaves: dict):
        from ..model.tiny_adapters import active

        return active([self.engine.resolve_adapters(adapters, leaves)])

    # Producers ------------------------------------------------------------------------------
    def _index_producers(self, producers: list) -> dict:
        """Block ID → (producer turn, its prompt items, its reply items, the block's index in the reply)."""
        index = {}
        for producer in producers or []:
            prompt, reply = self._target_items(producer.get("messages") or [], producer.get("tools"), producer["reply"])
            for at, (kind, value) in enumerate(reply):
                if kind == "block":
                    index.setdefault(value, (producer, prompt, reply, at))
        return index

    @staticmethod
    def _block_ids(items) -> list[str]:
        return [value for kind, value in items if kind == "block"]

    def _rewritten(self, block_id: str, leaves: dict, produced: dict, memo: dict, visiting: tuple = ()):
        """The block re-written from its producing turn with gradient, or None when that turn's context does not
        depend on an argument (the stored block is then exact and constant)."""
        if block_id in memo:
            return memo[block_id]
        from ..train.execution import unroll_write

        producer, prompt, reply, at = produced[block_id]
        context = prompt + reply[:at]
        local = {}
        for inner in dict.fromkeys(self._block_ids(context)):
            if inner in produced and inner not in leaves and inner not in visiting:
                value = self._rewritten(inner, leaves, produced, memo, visiting + (block_id,))
                if value is not None:
                    local[inner] = value
        adapters = producer.get("adapters") or []
        depends = bool(local) or any(b in leaves for b in self._block_ids(context)) or \
            any(a.get("id") in leaves for a in adapters if isinstance(a, dict))
        if not depends:
            memo[block_id] = None
            return None
        backbone, heads = self.backbone, self.heads
        stored = self.engine.lookup(block_id)
        with self._adapted(adapters, leaves):
            from ..train.execution import prefill_write_context
            pre = prefill_write_context(backbone, heads, self._embed_items(context, {**leaves, **local}))
            written = unroll_write(backbone, heads, pre, length=max(1, stored.length))
        rewritten = written.payload[0, :stored.length].float()
        recorded = stored.payload.clone().to(rewritten.device, rewritten.dtype)
        memo[block_id] = recorded + (rewritten - rewritten.detach())
        return memo[block_id]

    def _term_blocks(self, term: dict) -> list[str]:
        """Blocks a term reads: in its messages (and teacher messages), and in its target before the last."""
        found = []
        for key in ("messages", "teacher_messages"):
            messages = term.get(key)
            if messages:
                rendered = render_messages(messages, term.get("tools"), self.engine._template, self.engine.specials, block_type=self.engine.block_value_type)
                found.extend(rendered.blocks)
        target = term.get("target")
        if isinstance(target, dict) and term.get("messages"):
            _, reply = self._target_items(term["messages"], term.get("tools"), target)
            found.extend(self._block_ids(reply))
        return list(dict.fromkeys(found))

    def _term_value(self, term: dict, leaves: dict, produced: dict, adapters) -> torch.Tensor:
        """One weighted term; blocks it reads that a recorded turn wrote from an argument-dependent context are
        re-written with gradient for it."""
        term_leaves, memo = leaves, {}
        if produced:
            rewritten = {b: self._rewritten(b, leaves, produced, memo)
                         for b in self._term_blocks(term) if b in produced and b not in leaves}
            term_leaves = {**leaves, **{b: v for b, v in rewritten.items() if v is not None}}
        with self._adapted(term.get("adapters", adapters), leaves):
            return float(term.get("weight", 1.0)) * self._term(term, term_leaves).float().reshape(())

    def _derive(self, derived: list, values: dict) -> dict:
        """Second order: each inner computation recomputed as a function of `values` (the arguments, then every
        derived block before it). Returns `values` with the derived blocks that depend on them."""
        engine = self.engine
        values = dict(values)

        def tensor(block_id):
            if block_id in values:
                return values[block_id]
            block = engine.store.get(block_id)
            if block is None:
                raise RequestError("neuralese-unknown-block", block_id)
            return block.payload.clone().to(engine.device).float()

        def keep(block_id, value):
            """A derived value stands for its recorded block: the recorded value, the derived gradient."""
            if value is None or not value.requires_grad:
                return
            recorded = tensor(block_id).detach()
            values[block_id] = recorded + (value - value.detach())

        for step in derived:
            kind = step.get("kind")
            if kind == "grad":
                inner = list(dict.fromkeys(step.get("arguments") or []))
                inner_leaves = {b: values[b] if b in values else lookup_param(engine, b).payload.to(engine.device)
                                .clone().float().requires_grad_(True) for b in inner}
                scope = {**values, **inner_leaves}
                produced = self._index_producers(step.get("producers"))
                loss = sum(self._term_value(t, scope, produced, step.get("adapters")) for t in step.get("terms") or [])
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                parts = torch.autograd.grad(loss, list(inner_leaves.values()), create_graph=True, allow_unused=True)
                for block_id, part in zip(inner_leaves, parts):
                    gradient = (step.get("gradients") or {}).get(block_id)
                    if gradient and part is not None:
                        keep(gradient, part)
            elif kind == "optim":
                results = step.get("results") or {}
                outs, out_state = results.get("params") or [], results.get("state") or {}
                state = step.get("state") or {}
                for index, (param_id, grad_id) in enumerate(zip(step.get("params") or [], step.get("grads") or [])):
                    p, g = tensor(param_id), tensor(grad_id)
                    previous = {key: tensor(ids[index]) if index < len(ids := state.get(key) or []) and ids[index]
                                else torch.zeros_like(p) for key in ("m", "v")}
                    new_p, moments = functional_step(step.get("optimizer"), step.get("hyper") or {},
                                                     int(state.get("step") or 0) + 1, p, g, previous)
                    if index < len(outs):
                        keep(outs[index], new_p)
                    for key, value in moments.items():
                        ids = out_state.get(key) or []
                        if index < len(ids) and ids[index]:
                            keep(ids[index], value)
            else:
                raise RequestError("neuralese-grad-derived", f"unknown derived computation {kind!r}")
        return values

    def run(self, body: dict) -> dict:
        order = int(body.get("order") or 1)
        if order not in (1, 2):
            raise Unavailable("gradients of order 1 or 2")
        if order == 2:
            with second_order(self.backbone):
                return self._run(body, body.get("derived") or [])
        return self._run(body, [])

    def _run(self, body: dict, derived: list) -> dict:
        arguments = list(dict.fromkeys(body.get("arguments") or []))
        engine = self.engine
        leaves = {}
        for block_id in arguments:
            leaves[block_id] = lookup_param(engine, block_id).payload.to(engine.device).clone().float().requires_grad_(True)
        produced = self._index_producers(body.get("producers")) if leaves else {}
        terms = body.get("terms") or []
        if not terms:
            raise RequestError("neuralese-grad-term", "a grad request needs at least one term")
        arguments_only = leaves
        # Terms are differentiated one at a time and their gradients summed: the same gradient as the whole sum,
        # with the peak memory of the largest single term (one case's graph) rather than of all of them.
        grads = {block_id: torch.zeros_like(leaf) for block_id, leaf in leaves.items()}
        losses = []
        with torch.enable_grad():
            if derived and leaves:
                # The inner computations' graph is shared by every term: it is kept until the last term's backward.
                leaves = self._derive(derived, leaves)
            for index, term in enumerate(terms):
                # Blocks this term reads that a recorded turn wrote from an argument-dependent context: re-written
                # with gradient for this term (each term's graph is freed after its backward pass).
                value = self._term_value(term, leaves, produced, body.get("adapters"))
                if arguments_only and value.requires_grad:
                    parts = torch.autograd.grad(value, list(arguments_only.values()), allow_unused=True,
                                                retain_graph=bool(derived) and index < len(terms) - 1)
                    for (block_id, _), part in zip(arguments_only.items(), parts):
                        if part is not None:
                            grads[block_id] += part
                losses.append(value.detach())
                del value
        loss = torch.stack(losses).sum()
        out = {}
        for block_id, value in grads.items():
            source = engine.store.get(block_id)
            block = engine.store.put(make_block(value.detach(), grad_dialect(source.dialect), type=source.type,
                                                producer={"kind": "gradient", "of": block_id}))
            out[block_id] = block.id
        return {"loss": float(loss.detach()), "terms": [float(l.detach()) for l in losses], "gradients": out}


def decide(engine, body: dict) -> dict:
    """`POST /v1/neuralese/decide`: log-probabilities of finite replies (the runtime's decision readout)."""
    session = GradSession(engine)
    with torch.no_grad(), session._adapted(body.get("adapters"), {}):
        scores, tokens = session.decision_logprobs(body.get("messages") or [], body.get("tools"),
                                                               body.get("options") or [], {})
    return {"log_probs": [float(v) for v in scores], "tokens": tokens}


# Optimisers --------------------------------------------------------------------------------------
def optim_step(engine, body: dict) -> dict:
    """SGD (momentum) or Adam on argument blocks; returns new parameter and state blocks."""
    name = body.get("optimizer")
    hyper = body.get("hyper") or {}
    params, grads = body.get("params") or [], body.get("grads") or []
    if len(params) != len(grads):
        raise RequestError("neuralese-optim", "one gradient per parameter")
    state = body.get("state") or {}
    step = int(state.get("step") or 0) + 1
    new_params, new_state = [], {"step": step}
    moments = {"m": [], "v": []}
    for index, (param_id, grad_id) in enumerate(zip(params, grads)):
        param = lookup_param(engine, param_id)
        grad_block = engine.store.get(grad_id)
        if grad_block is None:
            raise RequestError("neuralese-unknown-block", grad_id)
        if grad_block.dialect != grad_dialect(param.dialect) or grad_block.payload.shape != param.payload.shape:
            raise RequestError("neuralese-optim", f"{grad_id} is not a gradient of {param_id}")
        p, g = param.payload.clone().float(), grad_block.payload.clone().float()

        def previous(key):
            ids = state.get(key) or []
            if index < len(ids) and ids[index]:
                return engine.store.get(ids[index]).payload.clone().float()
            return torch.zeros_like(p)

        p_new, step_moments = functional_step(name, hyper, step, p, g, {"m": previous("m"), "v": previous("v")})
        moments["m"].append(step_moments.get("m"))
        if "v" in step_moments:
            moments["v"].append(step_moments["v"])
        new_params.append(engine.store.put(make_block(p_new, param.dialect, type=param.type,
                                                      producer={"kind": "optim", "optimizer": name, "step": step,
                                                                "from": param_id})).id)
    for key, values in moments.items():
        if values and any(v is not None for v in values):
            new_state[key] = [None if v is None else engine.store.put(
                make_block(v, state_dialect(lookup_param(engine, params[i]).dialect),
                           producer={"kind": f"optim-{key}"})).id for i, v in enumerate(values)]
    return {"params": new_params, "state": new_state}


def functional_step(name, hyper: dict, step: int, p: torch.Tensor, g: torch.Tensor, previous: dict):
    """One optimiser step as a function of its tensors (differentiable in all of them): (new parameter, moments)."""
    lr = float(hyper.get("lr", 1e-3))
    weight_decay = float(hyper.get("weightDecay", 0.0))
    if name == "sgd":
        momentum = float(hyper.get("momentum", 0.0))
        m = momentum * previous["m"] + g if momentum else g
        return p - lr * (m + weight_decay * p), ({"m": m} if momentum else {})
    if name == "adam":
        beta1, beta2 = (float(b) for b in (hyper.get("betas") or (0.9, 0.999)))
        eps = float(hyper.get("eps", 1e-8))
        m = beta1 * previous["m"] + (1 - beta1) * g
        v = beta2 * previous["v"] + (1 - beta2) * g * g
        m_hat, v_hat = m / (1 - beta1 ** step), v / (1 - beta2 ** step)
        return p - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p), {"m": m, "v": v}
    raise RequestError("neuralese-optim", f"unknown optimizer {name!r}")


@contextlib.contextmanager
def second_order(backbone):
    """Kernels with a double backward: attention in its math kernel, the convolution without the fused kernel."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    fast = getattr(backbone, "fast", None)
    if fast is not None:
        backbone.fast = False
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        if fast is not None:
            backbone.fast = fast


def lookup_param(engine, block_id: str) -> Block:
    """A block that may be a parameter: a value in the server's dialect or an adapter for its backbone."""
    from ..model.tiny_adapters import is_adapter_dialect

    block = engine.store.get(block_id)
    if block is not None and is_adapter_dialect(block.dialect):
        engine.lookup_adapter(block_id)
        return block
    return engine.lookup(block_id)


def new_adapter(engine, body: dict) -> Block:
    """A zero adapter for this backbone (`POST /v1/neuralese/adapters`): `{"kind", "rank", "u", "layers",
    "targets", "seed"}`; layers default to those from the sketch cutoff to the top."""
    from ..model.tiny_adapters import AdapterSpec

    bank = engine.adapter_bank
    try:
        spec = bank.spec(kind=body.get("kind") or "xs", rank=int(body.get("rank") or 8), dim=int(body.get("u") or 0),
                         layers=body.get("layers"), targets=tuple(body.get("targets") or ("out", "ffn_down")),
                         seed=int(body.get("seed") or 0), cutoff=engine.heads.cutoff)
    except ValueError as error:
        raise RequestError("neuralese-adapter", str(error)) from error
    assert AdapterSpec.parse(spec.dialect()) == spec
    return engine.store.put(make_block(bank.zeros(spec), spec.dialect(), type=body.get("type") or "Adapter",
                                       producer={"kind": "adapter-init"}))


def embed_text(engine, text: str, type: str | None = None) -> Block:
    """A block initialised from text: the token embeddings of `text` (runtime registration, no writer)."""
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RequestError("neuralese-embed", "empty text")
    with torch.no_grad():
        rows = engine.backbone.embed(torch.tensor([ids], device=engine.device))[0].float()
    return engine.store.put(make_block(rows, engine.dialect, type=type, producer={"kind": "text-init", "text": text}))


def encode_text(engine, text: str, type: str | None = None, context: list | None = None) -> Block:
    """A block that encodes `text` in one forward pass: the port's write procedure with the text's token embeddings
    supplied at the sketch positions (the supplied-input regime of S3 phase A), so the transformer's output port turns
    each token into a payload vector. No sampling and no stop decision: one vector per token, no compression (the
    writer, with its stop head, compresses). The write site is `context` (chat messages, rendered with a generation
    prompt) or nothing, then the open marker, so purpose can come from context as for any write."""
    from ..train.execution import _complete, prefill, supplied_inputs

    backbone, heads = engine.backbone, engine.heads
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RequestError("neuralese-encode", "empty text")
    if context:
        prompt = render_messages(context, None, engine._template, engine.specials, block_type=engine.block_value_type)
        if prompt.blocks:
            raise RequestError("neuralese-encode", "the encoding context may not hold blocks")
        prefix = [i for segment in prompt.segments for i in engine._template_tokens(segment, prompt.escape_nonce)]
    else:
        # Raw ports write from a causal prefix, so a context-free encode starts at the document start: the tokenizer's
        # BOS, else the model config's (Qwen-family tokenizers, e.g. Maple, declare none but the config names one).
        bos = engine.tokenizer.bos_token_id
        if bos is None:
            bos = getattr(getattr(getattr(backbone, 'hf', None), 'config', None), 'bos_token_id', None)
        prefix = [bos] if bos is not None else []
    device = engine.device
    with torch.no_grad():
        pre = prefill(backbone, heads, torch.tensor([prefix + [backbone.controls.open_id]], device=device), logits=False)
        inputs = supplied_inputs(backbone, heads, torch.tensor([ids], device=device))
        shallow, _ = backbone.run_layers(inputs, range(0, heads.cutoff), pre.cache)
        _, sample = _complete(backbone, heads, pre.cache, inputs, shallow, top=pre.top)
    return engine.store.put(make_block(sample.payload[0].float(), engine.dialect, type=type,
                                       producer={"kind": "text-encode", "text": text}))


__all__ = ["GradSession", "Unavailable", "optim_step", "embed_text", "encode_text", "new_adapter", "lookup_param", "grad_dialect", "math"]
