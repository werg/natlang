"""Geometry-first, adaptive memory admission; no speculative model forwards."""
import math


class AdaptiveGraphMemory:
    def __init__(self, state=None, margin=.05):
        self.samples = dict((state or {}).get('samples', {}))
        self.margin = margin
        self.joint_ratios = dict((state or {}).get('joint_ratios', {}))

    @staticmethod
    def key(kind, context, length):
        return f'{kind}:{int(math.log2(max(1, context)))}:{int(math.log2(max(1, length)))}'

    def predict(self, kind, context, length, raw_bytes):
        samples = self.samples.get(self.key(kind, context, length), [])
        if len(samples) < 3:
            ratio = 1.0
        else:
            ordered = sorted(samples)
            ratio = ordered[min(len(ordered) - 1, int(.8 * len(ordered)))]
        return int(raw_bytes * max(.1, ratio) * (1 + self.margin))

    def observe(self, kind, context, length, raw_bytes, retained_bytes):
        if raw_bytes <= 0 or retained_bytes <= 0:
            return
        key = self.key(kind, context, length)
        self.samples[key] = (self.samples.get(key, []) + [retained_bytes / raw_bytes])[-64:]

    @staticmethod
    def joint_key(plan):
        count = len(plan['writers'])
        context = max([plan['reader_context']] + [c for c, _ in plan['writers']])
        return f"joint:{int(math.log2(max(1, count)))}:{int(math.log2(max(1, context)))}"

    def adjust_joint(self, plan, predicted):
        ratios = self.joint_ratios.get(self.joint_key(plan), [])
        if not ratios:
            return predicted
        ordered = sorted(ratios)
        ratio = ordered[min(len(ordered) - 1, int(.8 * len(ordered)))]
        return int(predicted * max(1.0, ratio))

    def observe_joint(self, plan, predicted, peak, *, failed=False):
        if predicted <= 0:
            return
        # Failed attempts are censored lower bounds, not successful peaks.
        # A small uplift makes the next related graph avoid the same miss.
        ratio = peak / predicted * (1.03 if failed else 1.0)
        key = self.joint_key(plan)
        self.joint_ratios[key] = (self.joint_ratios.get(key, []) + [ratio])[-64:]

    def state_dict(self):
        return {'samples': self.samples, 'joint_ratios': self.joint_ratios}


def geometry_bytes(context, vectors, *, width, layers, intermediate, kv_width, dtype_bytes,
                   checkpointed, target_tokens=0, vocab_size=0):
    # Checkpointed layers retain residual inputs; ordinary backward also retains
    # expanded FFN and operator activations. KV copies grow with sketch writes.
    tape_width = width if checkpointed else 6 * width + 3 * intermediate
    tape = context * layers * tape_width * dtype_bytes
    kv = context * kv_width * dtype_bytes * (1 + vectors)
    # Reader CE/KL retains logits/probability matrices in float32 as well as
    # native logits. Runtime observations refine this initial approximation.
    logits = target_tokens * vocab_size * 16
    return int(tape + kv + logits)
