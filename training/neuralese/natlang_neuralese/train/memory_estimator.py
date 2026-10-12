"""Geometry-first, adaptive memory admission; no speculative model forwards."""
import math

# Shared with memory_policy.choose_layer_checkpointing: a prediction below
# this count remains subject to its conservative uncalibrated multiplier.
CALIBRATED_SAMPLES = 3
DIRECT_SFT_ADJACENT_CONTEXT_DISTANCE = 1
DIRECT_SFT_ADJACENT_MIN_SAMPLES = 20
DIRECT_SFT_ADJACENT_QUANTILE = .95


def backbone_memory_layout(backbone, *, depth=None, checkpointed, **stage_options):
    """Build the activation geometry shared by trajectory and text warm-up.

    ``depth`` limits both layer count and summed K/V projection widths, which
    represents the actual shallow producer used by warm-up. Stage-specific
    geometry can be supplied as extra keyword options.
    """
    layers = int(backbone.num_layers)
    depth = layers if depth is None else int(depth)
    if not 0 <= depth <= layers:
        raise ValueError('backbone layout depth must be within the model')
    first = backbone.layers[0]
    if hasattr(first, 'feed_forward'):
        intermediate = int(first.feed_forward.w1.out_features)
    elif getattr(backbone.config, 'num_experts', 0):
        intermediate = (int(backbone.config.num_experts_per_tok) *
                        int(backbone.config.moe_intermediate_size))
    else:
        intermediate = int(first.mlp.gate_proj.out_features)
    kv_width = sum(2 * int(backbone.layers[i].self_attn.k_proj.out_features)
                   for i in range(depth) if backbone.is_attention(i))
    # RMSNorm implementations that promote their input to FP32 retain those
    # promoted values for backward when the layer is not checkpointed. Count
    # the actual call-site widths: attention q/k norms operate over all heads,
    # even when each norm's learned weight is only head_dim wide.
    width = int(backbone.config.hidden_size)
    norm_saved_width = 0
    norm_transient_width = 0
    activation_bytes = int(backbone.embedding_weight.element_size())
    if activation_bytes < 4:
        for i in range(depth):
            layer = backbone.layers[i]
            layer_widths = []
            for name in ('operator_norm', 'ffn_norm'):
                module = getattr(layer, name, None)
                if module is not None and 'rmsnorm' in type(module).__name__.casefold():
                    layer_widths.append(width)
            if backbone.is_attention(i):
                attention = layer.self_attn
                for norm_name, proj_name in (('q_layernorm', 'q_proj'), ('k_layernorm', 'k_proj')):
                    module = getattr(attention, norm_name, None)
                    projection = getattr(attention, proj_name, None)
                    if (module is not None and projection is not None and
                            'rmsnorm' in type(module).__name__.casefold()):
                        layer_widths.append(int(projection.out_features))
            norm_saved_width += sum(layer_widths)
            norm_transient_width = max(norm_transient_width, *layer_widths, 0)
    return {
        'width': int(backbone.config.hidden_size),
        'layers': depth,
        'intermediate': intermediate,
        'kv_width': kv_width,
        'dtype_bytes': activation_bytes,
        'fp32_norm_saved_bytes_per_token': norm_saved_width * 4,
        'fp32_norm_transient_bytes_per_token': norm_transient_width * 4,
        'checkpointed': bool(checkpointed),
        **stage_options,
    }


def writer_batch_kind(count, *, selective=False):
    """One namespace for admission and live writer-only tape observations."""
    return f"writer-batch{'-selective' if selective else ''}-tape-v2:{count}"


class ReplayResourceChoice:
    """Choose from live resource measurements once, then replay that exact path."""
    def __init__(self):
        self.value = None

    def resolve(self, choose):
        if self.value is None:
            self.value = bool(choose())
        return self.value


class AdaptiveGraphMemory:
    def __init__(self, state=None, margin=.05, geometry_version=None):
        old = state or {}
        previous = old.get('geometry_version', 'full-prefix-v1')
        self.geometry_version = geometry_version or previous
        # Adding a missing tape term changes the estimator, not actual joint
        # execution. Known failed route bounds remain valid in that case.
        self.joint_routes_compatible = (previous.removesuffix(':local-stage-geometry-v1').removesuffix(':native-gold-tape-v1') ==
                                        self.geometry_version.removesuffix(':local-stage-geometry-v1').removesuffix(':native-gold-tape-v1'))
        self.calibration_reset = previous != self.geometry_version
        self.reset_reason = ('geometry changed from ' + previous + ' to ' + self.geometry_version
                             if previous != self.geometry_version else old.get('reset_reason'))
        if self.calibration_reset:
            old = {}
        self.samples = dict(old.get('samples', {}))
        self.margin = margin
        # Ratios of the observed to the predicted variable part (graph tape) of a joint attempt. The fixed part
        # (resident weights, optimizer state, the update's gradient buffers) is counted exactly by the caller;
        # older whole-peak ratios ('joint_ratios') mixed both and are not reused.
        self.joint_ratios = dict(old.get('joint_tape_ratios', {}))

    @staticmethod
    def key(kind, context, length):
        return f'{kind}:{int(math.log2(max(1, context)))}:{int(math.log2(max(1, length)))}'

    def calibration(self, kind, context, length):
        exact_key = self.key(kind, context, length)
        samples = self.samples.get(exact_key, [])
        # Direct SFT's raw geometry includes the target-length-specific readout
        # workspace (capped at 128 positions). Pool sparse target bins only
        # within the same execution/checkpoint family and context bin: this
        # reuses observed layer activation ratios while the raw denominator
        # still accounts for each target's own readout size.
        if len(samples) < 3 and kind.startswith('direct-sft-'):
            context_bin = int(math.log2(max(1, context)))
            family_prefix = f'{kind}:{context_bin}:'
            pooled = []
            for key, values in self.samples.items():
                if not key.startswith(family_prefix):
                    continue
                if not key.rsplit(':', 1)[-1].isdigit():
                    continue
                pooled.extend(values)
            if len(pooled) >= 3:
                samples = pooled
        # Batch geometry already includes the row count. Reuse only current
        # writer-only measurements, with identical policy and geometry bins.
        # Old observations included released auxiliary tapes and are excluded.
        if len(samples) < 3 and kind.startswith(('writer-batch-tape-v2:', 'writer-batch-selective-tape-v2:')):
            family = kind.rsplit(':', 1)[0] + ':'
            suffix = ':' + self.key('', context, length).split(':', 1)[1]
            siblings = [values for key, values in self.samples.items()
                        if key.startswith(family) and key.endswith(suffix) and len(values) >= 3]
            if siblings:
                samples = samples + [ratio for values in siblings for ratio in values]
        return samples

    def _calibration_for_prediction(self, kind, context, length):
        """Select matching observations, optionally seeding sparse direct-SFT context bins.

        The adjacent-bin fallback only applies to the same direct-SFT execution
        and checkpoint kind and geometry namespace. It uses one adjacent context
        bin and needs twenty observations. Target bins at that source context
        may pool because raw geometry already includes the requested target's
        own readout workspace. Direct-SFT estimator state is stored within an
        exact-resume checkpoint whose model and layout identity is verified by
        the caller; it must not be detached and reused for another model/layout.
        """
        exact_key = self.key(kind, context, length)
        exact = self.samples.get(exact_key, [])
        if len(exact) >= CALIBRATED_SAMPLES:
            return exact, {'sample_count': len(exact), 'source_bins': [exact_key],
                           'source': 'exact-target-bin', 'context_bin_distance': 0}
        context_bin = int(math.log2(max(1, context)))
        family_prefix = f'{kind}:{context_bin}:'
        if kind.startswith('direct-sft-'):
            same_context = [(key, values) for key, values in self.samples.items()
                            if key.startswith(family_prefix) and
                            key.rsplit(':', 1)[-1].isdigit() and values]
            pooled = [ratio for _, values in same_context for ratio in values]
            if len(pooled) >= CALIBRATED_SAMPLES:
                return pooled, {'sample_count': len(pooled),
                                'source_bins': sorted(key for key, _ in same_context),
                                'source': 'same-context-target-pool', 'context_bin_distance': 0}
        else:
            pooled = self.calibration(kind, context, length)
            if len(pooled) >= CALIBRATED_SAMPLES:
                return pooled, {'sample_count': len(pooled), 'source_bins': [exact_key],
                                'source': 'existing-calibration', 'context_bin_distance': 0}
        if kind.startswith('direct-sft-'):
            candidates = {}
            for key, values in self.samples.items():
                pieces = key.rsplit(':', 2)
                if (len(pieces) != 3 or pieces[0] != kind or
                        not pieces[1].isdigit() or not pieces[2].isdigit() or not values):
                    continue
                source_context_bin = int(pieces[1])
                distance = abs(source_context_bin - context_bin)
                if 0 < distance <= DIRECT_SFT_ADJACENT_CONTEXT_DISTANCE:
                    candidates.setdefault(source_context_bin, []).append((key, values))
            eligible = [(abs(source_bin - context_bin), -sum(len(v) for _, v in entries), source_bin, entries)
                        for source_bin, entries in candidates.items()
                        if sum(len(v) for _, v in entries) >= DIRECT_SFT_ADJACENT_MIN_SAMPLES]
            if eligible:
                distance, _, source_bin, entries = min(eligible)
                source_bins = sorted(key for key, _ in entries)
                samples = [ratio for _, values in entries for ratio in values]
                return samples, {'sample_count': len(samples), 'source_bins': source_bins,
                                 'source': 'adjacent-context',
                                 'context_bin_distance': distance,
                                 'source_context_bin': source_bin}
        return exact, {'sample_count': len(exact), 'source_bins': [exact_key],
                       'source': 'uncalibrated', 'context_bin_distance': None}

    def calibration_info(self, kind, context, length):
        """Return the evidence selected for a prediction, including its source bins."""
        _, details = self._calibration_for_prediction(kind, context, length)
        # Exact bins are not intrinsically safer than adjacent bins: sparse
        # q80 estimates underestimated a real direct-SFT readout peak.
        details['quantile'] = (DIRECT_SFT_ADJACENT_QUANTILE
                               if kind.startswith('direct-sft-') else .8)
        return details

    def predict(self, kind, context, length, raw_bytes):
        samples, details = self._calibration_for_prediction(kind, context, length)
        if len(samples) < 3:
            ratio = 1.0
        else:
            ordered = sorted(samples)
            if kind.startswith('direct-sft-'):
                rank = max(0, math.ceil(DIRECT_SFT_ADJACENT_QUANTILE * len(ordered)) - 1)
            else:
                # Preserve the established q80 selection for existing bins.
                rank = int(.8 * len(ordered))
            ratio = ordered[min(len(ordered) - 1, rank)]
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

    def joint_ratio(self, plan):
        """The 80th-percentile tape ratio of this plan's bin; a bin not seen yet borrows every bin's observations
        (a first two-writer record must not start uncalibrated after a one-writer record missed by 2x)."""
        ratios = self.joint_ratios.get(self.joint_key(plan)) or [r for values in self.joint_ratios.values() for r in values]
        if not ratios:
            return 1.0
        ordered = sorted(ratios)
        return max(1.0, ordered[min(len(ordered) - 1, int(.8 * len(ordered)))])

    def adjust_joint(self, plan, predicted, fixed=0):
        """``predicted`` is the whole raw peak (fixed + tape); only the tape above ``fixed`` is scaled."""
        tape = max(0, predicted - fixed)
        return int(fixed + tape * self.joint_ratio(plan))

    def observe_joint(self, plan, predicted, peak, *, failed=False, fixed=0):
        tape = predicted - fixed
        if tape <= 0:
            return None
        # Failed attempts are censored lower bounds, not successful peaks.
        # A small uplift makes the next related graph avoid the same miss.
        ratio = max(0.0, peak - fixed) / tape * (1.03 if failed else 1.0)
        key = self.joint_key(plan)
        self.joint_ratios[key] = (self.joint_ratios.get(key, []) + [ratio])[-64:]
        return ratio

    def state_dict(self):
        return {'samples': self.samples, 'joint_tape_ratios': self.joint_ratios,
                'geometry_version': self.geometry_version, 'reset_reason': self.reset_reason}


def geometry_bytes(context, vectors, *, width, layers, intermediate, kv_width, dtype_bytes,
                   checkpointed, target_tokens=0, vocab_size=0, shared_kv_prefix=False,
                   uncheckpointed_layers=0, stage_group_size=0, sketch_cutoff=0,
                   fp32_norm_saved_bytes_per_token=0, fp32_norm_transient_bytes_per_token=0):
    # Checkpointed layers retain residual inputs; ordinary backward also retains
    # expanded FFN and operator activations. KV copies grow with sketch writes.
    plain_layers = uncheckpointed_layers if checkpointed else layers
    if not 0 <= plain_layers <= layers:
        raise ValueError('invalid uncheckpointed layer count')
    tape_per_token = ((layers - plain_layers) * width +
                      plain_layers * (6 * width + 3 * intermediate)) * dtype_bytes
    tape = context * tape_per_token
    if min(fp32_norm_saved_bytes_per_token, fp32_norm_transient_bytes_per_token) < 0:
        raise ValueError('FP32 norm geometry must be nonnegative')
    if layers:
        tape += context * fp32_norm_saved_bytes_per_token * plain_layers // layers
    if plain_layers < layers:
        # Checkpointing discards per-call saved tensors, but one promoted norm
        # input remains live transiently while the current layer executes.
        tape += context * fp32_norm_transient_bytes_per_token
    if stage_group_size and vectors:
        if stage_group_size < 1 or not 0 <= sketch_cutoff <= layers:
            raise ValueError('invalid local-stage geometry')
        groups = [min(stage_group_size, vectors - start) for start in range(0, vectors, stage_group_size)]
        # Each branch is [group rows,group positions]. Fixed history runs the
        # shallow layers for all positions; upper history excludes the last group.
        layer_tokens = sum(g * g for g in groups) * layers
        layer_tokens += vectors * sketch_cutoff + (vectors - groups[-1]) * (layers - sketch_cutoff)
        tape += layer_tokens * tape_per_token / layers
    positions = (context + vectors * (vectors + 1) // 2
                 if shared_kv_prefix else context * (1 + vectors))
    kv = positions * kv_width * dtype_bytes
    # Reader CE/KL retains logits/probability matrices in float32 as well as
    # native logits. Runtime observations refine this initial approximation.
    logits = target_tokens * vocab_size * 16
    return int(tape + kv + logits)


def selective_writer_fits(context, vectors, layout, *, baseline, budget, plain_layers):
    """Admit extra convolution tapes from geometry, without speculative forwards."""
    if not plain_layers or baseline >= budget:
        return False
    estimate = geometry_bytes(context, vectors, **{**layout, 'uncheckpointed_layers': plain_layers})
    return baseline + estimate <= budget


def producer_geometry_bytes(context, vectors, layout, *, native_gold=False, vocab_size=0):
    """Joint producer retains both generated writer and native gold CE tapes.

    Staging releases these separately; joint admission must count both.
    The gold pass scores the forced prefix plus every body token, including
    logits even for stop-only supervision (the scoring path computes them).
    """
    raw = geometry_bytes(context, vectors, **layout)
    if native_gold:
        raw += geometry_bytes(context + vectors, 0, **layout,
                              target_tokens=vectors, vocab_size=vocab_size)
    return raw


def local_stage_kv_workspace(context, vectors, group_size, *, max_layer_kv_width, dtype_bytes):
    """Largest one-layer concatenated branch K/V pair, separate from saved tape.

    Expanded prefix views share storage, but attention materializes their concat
    with the branch tail. This is a shape-derived lower workspace estimate; it
    does not assert that backend GQA/logit/FFN workspaces are fully accounted for.
    Joint plans take the largest workspace, not the sum over sequential writers.
    """
    if not group_size or not vectors:
        return 0
    return max(min(group_size, vectors-start) * (context+start+min(group_size, vectors-start))
               for start in range(0, vectors, group_size)) * max_layer_kv_width * dtype_bytes
