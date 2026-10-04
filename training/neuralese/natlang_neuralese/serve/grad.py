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
| `decision` | A proper scoring rule on the decision readout: `options` (assistant replies) are scored after `messages` as in `/v1/neuralese/decide`, normalised over the options, and compared with `target` probabilities (`rule`: `logLoss`, the default, is cross-entropy against the target distribution; `brier` is the squared error; `rps` is the ranked probability score for ordered options). |

Every term has a `weight`; the session loss is the weighted sum. Constants (`stopGradient`, inner gradients of a
first-order nested `grad`) are simply blocks that are not arguments. Exact second order is not supported.

`optim_step` applies SGD (with momentum) or Adam to argument blocks with their gradients and returns new parameter
blocks and new optimiser-state blocks: nothing is updated in place.
"""

from __future__ import annotations

import math

import torch

from ..model.heads import PayloadSample, payload_kl, payload_log_prob
from ..model.lfm2_port import PortCache
from .chat import RequestError, render_messages
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
    def _items(self, segments, blocks: list[str]) -> list:
        items = []
        for segment in segments:
            if isinstance(segment, str):
                items.extend(("tok", i) for i in self.engine._template_tokens(segment))
            else:
                items.append(("block", blocks[segment]))
        return items

    def _target_items(self, messages, tools, target) -> tuple[list, list]:
        """(prompt items, target items) with the target teacher-forced after the generation prompt."""
        engine = self.engine
        prompt = render_messages(messages, tools, engine._template, engine.specials)

        def full_template(msgs, tls):
            return engine.tokenizer.apply_chat_template(msgs, tools=tls or None, tokenize=False,
                                                        add_generation_prompt=False)

        full = render_messages(list(messages) + [target], tools, full_template, engine.specials)
        before = self._items(prompt.segments, prompt.blocks)
        after = self._items(full.segments, full.blocks)
        # The prompt is scored exactly as inference rendered it; the template may render earlier turns differently
        # once another assistant turn follows (it drops past reasoning), so the target is cut from the full rendering
        # after its own generation prefix rather than by matching the whole prompt.
        im_start = engine.tokenizer.convert_tokens_to_ids("<|im_start|>")
        starts = [i for i, item in enumerate(before) if item == ("tok", im_start)]
        if not starts:
            raise RequestError("neuralese-grad-target", "the rendered prompt has no generation prefix")
        prefix = before[starts[-1]:]
        found = [i for i in range(len(after) - len(prefix), -1, -1) if after[i:i + len(prefix)] == prefix]
        if not found:
            raise RequestError("neuralese-grad-target", "the target's rendering has no assistant prefix")
        rest = after[found[0] + len(prefix):]
        im_end = engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ends = [i for i, item in enumerate(rest) if item == ("tok", im_end)]
        if ends:
            rest = rest[:ends[-1] + 1]
        return before, rest

    def _payload(self, block_id: str, leaves: dict) -> torch.Tensor:
        if block_id in leaves:
            return leaves[block_id]
        return self.engine.lookup(block_id).payload.clone().to(self.engine.device)  # stored rows may be inference tensors

    def _embed_items(self, items, leaves) -> torch.Tensor:
        backbone, heads, device = self.backbone, self.heads, self.engine.device
        dtype = backbone.embedding_weight.dtype
        pieces, run = [], []

        def flush():
            if run:
                pieces.append(backbone.embed(torch.tensor([run], device=device)))
                run.clear()

        for kind, value in items:
            if kind == "tok":
                run.append(value)
            else:
                flush()
                pieces.append(backbone.embed(torch.tensor([[backbone.controls.open_id]], device=device)))
                pieces.append(heads.interface(self._payload(value, leaves).to(dtype))[None])
                pieces.append(backbone.embed(torch.tensor([[backbone.controls.close_id]], device=device)))
        flush()
        return torch.cat(pieces, 1)

    def _score(self, prompt, target, leaves, write_terms: bool) -> dict:
        """Teacher-forced pass over prompt + target. Returns per-position text log-probs (and logits) for the
        target's text tokens, plus write terms for written blocks."""
        backbone, heads = self.backbone, self.heads
        embeds = self._embed_items(prompt, leaves)
        out = backbone.forward_embeds(embeds)
        cache, last = out["cache"], out["logits"][:, -1]
        token_logp, token_logits, write_logp = [], [], []
        index = 0
        while index < len(target):
            kind, value = target[index]
            if kind == "tok":
                run = []
                while index < len(target) and target[index][0] == "tok":
                    run.append(target[index][1])
                    index += 1
                ids = torch.tensor([run], device=self.engine.device)
                step = backbone.forward_ids(ids, cache=cache)
                logits = torch.cat([last[:, None], step["logits"][:, :-1]], 1)[0]
                token_logits.append(logits)
                token_logp.append(torch.log_softmax(logits.float(), -1).gather(1, ids[0][:, None])[:, 0])
                cache, last = step["cache"], step["logits"][:, -1]
                continue
            # A written block: the open decision is a text decision; the write is replayed with its recorded length.
            open_id = backbone.controls.open_id
            token_logits.append(last)
            token_logp.append(torch.log_softmax(last.float(), -1)[:, open_id])
            block = self.engine.lookup(value)
            opened = backbone.forward_ids(torch.tensor([[open_id]], device=self.engine.device), cache=cache,
                                          cutoff=heads.cutoff)
            block_start, state = opened["cache"], opened["h_cut"][:, -1]
            if write_terms:
                write_logp.append(self._replay_write(block, block_start, state))
            payload = self._payload(value, leaves)[None].to(backbone.embedding_weight.dtype)
            close = torch.full((1, 1), backbone.controls.close_id, device=self.engine.device)
            back = backbone.forward_embeds(torch.cat([heads.interface(payload), backbone.embed(close)], 1),
                                           cache=block_start)
            cache, last = back["cache"], back["logits"][:, -1]
            index += 1
        return {"token_logp": torch.cat(token_logp) if token_logp else torch.zeros(0),
                "token_logits": torch.cat(token_logits) if token_logits else None,
                "write_logp": torch.stack(write_logp).sum() if write_logp else torch.zeros(())}

    def _replay_write(self, block: Block, block_start: PortCache, state: torch.Tensor) -> torch.Tensor:
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
            mu, log_sigma = heads.content.distribution(torch.stack(sketches, 1), final)
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
        out = self.backbone.forward_embeds(self._embed_items(prompt, leaves))
        cache, last = out["cache"], out["logits"][:, -1]
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

    def decision_backward(self, messages, tools, options: list[str], goal: torch.Tensor, scale: float = 1.0) -> float:
        """Backpropagate the cross-entropy of the normalised option distribution against `goal`, holding one option's
        graph at a time: scores are computed without gradient, then each option is re-run and backpropagated with
        its coefficient softmax_i - goal_i (the gradient of the loss in its score). Returns the loss."""
        with torch.no_grad():
            scores, _ = self.decision_logprobs(messages, tools, options, {})
        goal = goal / goal.sum()
        logp = torch.log_softmax(scores, 0)
        coefficients = (logp.exp() - goal) * scale
        cache, last, owns = self.decision_prepare(messages, tools, options, {})
        for index, own in enumerate(owns):
            if float(coefficients[index]) == 0.0:
                continue
            (coefficients[index] * self.option_logprob(cache, last, own)).backward(retain_graph=True)
        # The prompt's graph is shared by every option; release it after the last option.
        del cache, last
        return float(-(goal * logp).sum())

    def _decision_term(self, term: dict, leaves: dict) -> torch.Tensor:
        options, target = term.get("options") or [], term.get("target")
        probabilities = target.get("probabilities") if isinstance(target, dict) else None
        if not isinstance(probabilities, list) or len(probabilities) != len(options):
            raise RequestError("neuralese-grad-term", "decision needs target.probabilities, one per option")
        scores, _ = self.decision_logprobs(term.get("messages") or [], term.get("tools"), options, leaves)
        logp = torch.log_softmax(scores, 0)
        goal = torch.tensor([float(p) for p in probabilities], device=logp.device)
        goal = goal / goal.sum()
        rule = term.get("rule") or "logLoss"
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
            with torch.no_grad():
                t_prompt, t_rest = self._target_items(teacher_messages, tools, target)
                teacher = self._score(t_prompt, t_rest, {}, write_terms=False)["token_logits"]
            if student is None or teacher is None or student.shape != teacher.shape:
                raise RequestError("neuralese-grad-term", "student and teacher targets do not align")
            t = torch.log_softmax(teacher.float(), -1)
            s = torch.log_softmax(student.float(), -1)
            return (t.exp() * (t - s)).sum(-1).mean()
        raise RequestError("neuralese-grad-term", f"unknown term kind {kind!r}")

    def run(self, body: dict) -> dict:
        if int(body.get("order") or 1) != 1:
            raise Unavailable("only first-order gradients are supported")
        arguments = list(dict.fromkeys(body.get("arguments") or []))
        engine = self.engine
        leaves = {}
        for block_id in arguments:
            leaves[block_id] = engine.lookup(block_id).payload.to(engine.device).clone().float().requires_grad_(True)
        terms = body.get("terms") or []
        if not terms:
            raise RequestError("neuralese-grad-term", "a grad request needs at least one term")
        # Terms are differentiated one at a time and their gradients summed: the same gradient as the whole sum,
        # with the peak memory of the largest single term (one case's graph) rather than of all of them.
        grads = {block_id: torch.zeros_like(leaf) for block_id, leaf in leaves.items()}
        losses = []
        with torch.enable_grad():
            for term in terms:
                value = float(term.get("weight", 1.0)) * self._term(term, leaves).float().reshape(())
                if leaves and value.requires_grad:
                    parts = torch.autograd.grad(value, list(leaves.values()), allow_unused=True)
                    for (block_id, _), part in zip(leaves.items(), parts):
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
    with torch.no_grad():
        scores, tokens = GradSession(engine).decision_logprobs(body.get("messages") or [], body.get("tools"),
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
    lr = float(hyper.get("lr", 1e-3))
    new_params, new_state = [], {"step": step}
    moments = {"m": [], "v": []}
    for index, (param_id, grad_id) in enumerate(zip(params, grads)):
        param = engine.lookup(param_id)
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

        weight_decay = float(hyper.get("weightDecay", 0.0))
        if name == "sgd":
            momentum = float(hyper.get("momentum", 0.0))
            m = momentum * previous("m") + g if momentum else g
            p_new = p - lr * (m + weight_decay * p)
            moments["m"].append(m if momentum else None)
        elif name == "adam":
            beta1, beta2 = (float(b) for b in (hyper.get("betas") or (0.9, 0.999)))
            eps = float(hyper.get("eps", 1e-8))
            m = beta1 * previous("m") + (1 - beta1) * g
            v = beta2 * previous("v") + (1 - beta2) * g * g
            m_hat, v_hat = m / (1 - beta1 ** step), v / (1 - beta2 ** step)
            p_new = p - lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * p)
            moments["m"].append(m)
            moments["v"].append(v)
        else:
            raise RequestError("neuralese-optim", f"unknown optimizer {name!r}")
        new_params.append(engine.store.put(make_block(p_new, param.dialect, type=param.type,
                                                      producer={"kind": "optim", "optimizer": name, "step": step,
                                                                "from": param_id})).id)
    for key, values in moments.items():
        if values and any(v is not None for v in values):
            new_state[key] = [None if v is None else engine.store.put(
                make_block(v, state_dialect(engine.dialect), producer={"kind": f"optim-{key}"})).id for v in values]
    return {"params": new_params, "state": new_state}


def embed_text(engine, text: str, type: str | None = None) -> Block:
    """A block initialised from text: the token embeddings of `text` (runtime registration, no writer)."""
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RequestError("neuralese-embed", "empty text")
    with torch.no_grad():
        rows = engine.backbone.embed(torch.tensor([ids], device=engine.device))[0].float()
    return engine.store.put(make_block(rows, engine.dialect, type=type, producer={"kind": "text-init", "text": text}))


__all__ = ["GradSession", "Unavailable", "optim_step", "embed_text", "grad_dialect", "math"]
