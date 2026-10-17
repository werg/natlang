#!/usr/bin/env python3
"""Verify an explicitly reviewed optimizer or CPU-metadata capacity continuation.

This does not run a model and never claims a new parent was memory-qualified.
The independently approved review is policy; this tool verifies its exact bounds.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import json
import math
import re
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_resident import binding, bound as _resident_bound, code_inventory, same_geometry_identity

_DEPENDENCY_SINK=ContextVar('native_capacity_dependencies',default=None)


def bound(reference):
    path=_resident_bound(reference)
    sink=_DEPENDENCY_SINK.get()
    if sink is not None:sink[str(path)]=binding(path)
    return path

ALLOWED_CHANGES = {
    'train/native_optimizer_migration.py', 'train/native_semantic_stage.py',
    'train/native_state_sft.py', 'train/optim.py', 'train/optim_restore.py',
    'train/recipe.py',
}
POLICY = 'native-ordinary-to-adamw-new-history-v1'
METADATA_POLICY = 'native-scheduling-pending-obligations-v1'
METADATA_CHANGES = {
    'train/native_state_sft.py', 'train/recipe.py', 'train/native_source_streams.py',
    'train/native_cohort_schedule.py', 'train/native_operation_report.py',
    'train/native_pending_sources.py', 'train/native_continuation.py',
    'train/native_readiness.py', 'train/native_resident.py',
    'train/native_source_preparation.py', 'data/native_source_tree.py',
    'rl/native_supervised.py', 'rl/production.py',
}
LOADER_IO_POLICY = 'native-resident-mmap-advice-v1'
LOADER_IO_CHANGES = METADATA_CHANGES | {
    'train/checkpoint_policy.py', 'train/checkpoint_readahead.py',
}
LINEAGE_METADATA_POLICY = 'native-saved-recipe-lineage-metadata-v1'
LINEAGE_METADATA_FILE = 'train/native_state_sft.py'
LINEAGE_METADATA_COMPOSITION = (
    'ca5acf17bbcdfd0eaeff2712e9af07a990960d8235c6ab5bb89ae2625d04321e',
    '14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8',
)
POLICY_GATE_SFT_POLICY = 'native-frozen-policy-gate-ordinary-sft-v1'
POLICY_GATE_SFT_COMPOSITION = {
    'eval/native_memory.py': ('69885bd62fdfd4d2bdc73e5d4ac2f3c41aa4a006ecc4a72bd8125b1b2735d392', '05b23b60f54795f48261515022ee4fc6b01eedd5e87bbfcdfec98f3510e7dbeb'),
    'rl/native_replay.py': ('976e066cbe4103fd67a75bf97547a9a2b5f12398223a28e9686330e27522b0e2', '99340bac53db80afdc8fcc3d8f2d9beba73c42ad0a2813857ef80cd2abbd78dd'),
    'rl/native_stage.py': ('6dfa075691aae40bb97a7f5ed495c7ea1f29b257f05072fa1e217224840c4ac8', '9e8559a9e96c4ed94194a6b3bf0c9803b308d475380e0485f9b9c6e2c3997f8f'),
    'rl/production.py': ('207e44e05ccd970819695fddec66851fea8ee7c7d46acb11f7f5f3051c18e7f4', '6a02e01c1b0d623001ed19c34baed867ec456b74a89dcdf4f0e952cca99288bc'),
    'train/native_state_sft.py': ('14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8', 'c2b1688724eed94117f2ea1019bd7354579f5aeaeec0390719dac051e0fc461f'),
}


ORDINARY_WRITER_POLICY = 'native-frozen-ordinary-sft-report-writer-v1'
ORDINARY_WRITER_COMPOSITION = {
    'train/native_state_sft.py': (
        '14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8',
        'dd38e191c06500435e3cff556c09e3bfe7aa88b2460c0232f5b4221dc283cceb'),
    'train/native_operation_report.py': (None,
        '9243caf33e1a2e015c4cfca99c1600c418eaf339fa3900b3c19e4e756908a633'),
}


WHOLE_HEAD_SOURCE_POLICY = 'native-whole-head-source-profile-ordinary-sft-v1'
WHOLE_HEAD_SOURCE_PROOF_SHA256 = 'b7e28c394e26a6426c015e056b2c345c83616a31a382b078ab00e78d550e5387'
WHOLE_HEAD_SOURCE_REVIEWED = {
    'ac7575cc082193d8bc427c1d403637cbcb6d2f1fd1677f983edcd8d28d64a0f9': '7e3f359455dfc71c0876e9b0ddc03b6d8815594fd5aaa7d0fc6820e5c335d73d',
    'd885adc940a356d3f8cd012b0aa35a0a8357ead83a598d73626cfe5b5a979fe0': '7ac5641ff170ed75b67d4416d57e29631a3d9889c89d4a32baa2ba380080a9fc',
    '0ec7a87108f19f4b0fc44898b3cbb2031547f33b60d5ba0389bae07b8e720236': WHOLE_HEAD_SOURCE_PROOF_SHA256,
}
WHOLE_HEAD_SOURCE_INVENTORIES = ('42cbc4cce6df3a36d244767e3aae023c872daf0ee437ff6c05390e4670d31210', '0ec7a87108f19f4b0fc44898b3cbb2031547f33b60d5ba0389bae07b8e720236')
WHOLE_HEAD_SOURCE_CHANGES = frozenset(['data/native_source_tree.py', 'data/native_sources.py', 'eval/native_memory.py', 'eval/task_cadence.py', 'maple/model.py', 'model/native_bundle.py', 'model/native_composite.py', 'model/native_model.py', 'model/native_parameter_precision.py', 'model/native_precision.py', 'model/native_shared_checkpoint.py', 'model/shared_state.py', 'rl/native_bootstrap_metadata.py', 'rl/native_raw_handoff.py', 'rl/native_raw_policy.py', 'rl/native_raw_rounds.py', 'rl/native_refresh.py', 'rl/native_replay.py', 'rl/native_rounds.py', 'rl/native_runtime_handoff.py', 'rl/native_stage.py', 'rl/native_supervised.py', 'rl/production.py', 'rl/task_source.py', 'serve/engine.py', 'serve/http.py', 'serve/native_protocol.py', 'serve/native_recorded_objective.py', 'train/fp8_backend.py', 'train/native_operation_report.py', 'train/native_plan_recovery.py', 'train/native_readiness.py', 'train/native_resident_geometry_refresh.py', 'train/native_semantic.py', 'train/native_semantic_stage.py', 'train/native_source_preparation.py', 'train/native_state_sft.py', 'train/recipe.py'])


# The 378ff pair additionally changes metadata collection/completion plumbing.
# Full inventory and witness pins below still prohibit arbitrary edits here.
WHOLE_HEAD_SOURCE_CHANGES |= {'train/native_continuation.py'}


def _whole_head_source_contract(review):
    if review['optimizer_policy'] != WHOLE_HEAD_SOURCE_POLICY:
        return
    identity = review.get('identity') or {}
    config = identity.get('model_config') or {}
    if (identity.get('objective') is not None
            or review.get('fresh_policy_capacity_required') is not True
            or config.get('native_norm_execution', 'source-v1') != 'source-v1'
            or config.get('native_attention_precision', 'ieee') != 'ieee'
            or config.get('native_ordinary_precision') is not None):
        raise ValueError('whole HEAD continuation preserves ordinary source-v1/IEEE/storage; new profiles and policy require actual capacity')
    old_digest,new_digest = (digest(review[key]) for key in ('old_code','new_code'))
    if old_digest != WHOLE_HEAD_SOURCE_INVENTORIES[0] or new_digest not in WHOLE_HEAD_SOURCE_REVIEWED:
        raise ValueError('whole HEAD continuation differs from the exact independently reviewed full inventories')
    reference = review.get('whole_head_arithmetic_proof') or {}
    if reference.get('sha256') != WHOLE_HEAD_SOURCE_REVIEWED[new_digest]:
        raise ValueError('whole HEAD continuation lacks the exact arithmetic and source-parser witness')
    proof = json.loads(bound(reference).read_text())
    if (proof.get('schema') != 'natlang.whole-main-source-arithmetic-proof/1'
            or proof.get('inventories', {}).get('source') != review['old_code']
            or proof.get('inventories', {}).get('target') != review['new_code']
            or proof.get('remaining_unconditional_normal_arithmetic_changes') != []
            or proof.get('capacity_qualified') is not False):
        raise ValueError('whole HEAD source witness differs from the reviewed arithmetic scope')


def _ordinary_writer_contract(review):
    if review['optimizer_policy'] != ORDINARY_WRITER_POLICY:
        return
    if (review.get('identity', {}).get('objective') is not None
            or review.get('fresh_policy_capacity_required') is not True):
        raise ValueError('report-writer continuation is ordinary SFT only; actual policy capacity remains required')
    changes = {name for name in review['old_code'].keys() | review['new_code'].keys()
               if review['old_code'].get(name) != review['new_code'].get(name)}
    if (changes != set(ORDINARY_WRITER_COMPOSITION)
            or any((review['old_code'].get(name), review['new_code'].get(name)) != pair
                   for name, pair in ORDINARY_WRITER_COMPOSITION.items())):
        raise ValueError('report-writer continuation requires the exact two-file frozen ordinary SFT composition')


def _policy_gate_sft_contract(review):
    if review['optimizer_policy'] != POLICY_GATE_SFT_POLICY:
        return
    if (review.get('identity', {}).get('objective') is not None
            or review.get('fresh_policy_capacity_required') is not True):
        raise ValueError('policy-gate continuation is ordinary SFT only; actual policy capacity remains required')
    changes = {name for name in review['old_code'].keys() | review['new_code'].keys()
               if review['old_code'].get(name) != review['new_code'].get(name)}
    if (changes != set(POLICY_GATE_SFT_COMPOSITION)
            or any((review['old_code'].get(name), review['new_code'].get(name)) != pair
                   for name, pair in POLICY_GATE_SFT_COMPOSITION.items())):
        raise ValueError('ordinary SFT continuation requires the exact five-file frozen policy-gate composition')


ATTENTION_INDEX_POLICY = 'native-graph-attention-index-reuse-v1'
ATTENTION_INDEX_REVIEW_SHA256 = 'e100c0814676e340b6c2bf3341dee101d02958a249757304ca3fb2df16a0b7a2'
ATTENTION_INDEX_COMPOSITIONS = {
    'lfm': ('14ca1758cacf99a4dc8f237fea7b8005d2123c40',
            '21d878c9c8e7b1345d813b0827b05513cf8c9b55',
            'cac5df15b82939c4d618e70cdf1b82db00dd2fbf66a73e0ee5a3985eb6ee369d',
            '7cb9ab4fb9ea7e4c882f9021c0dc81507e33c4ecfde1e85ff4024f58c3644693'),
    'mellum': ('b50e8823aaa1f92d2d3be853afc1955948a8fd1f',
               '92c7b4ae03d13baab1d88f83df20c0b185f9439e',
               '0dd5044fe1a3d232c4160edbe9da92240c18f3341178d0f252503043bae5078a',
               'd663389095b4fe6fa4d6ec991f2a9c6f47697995f4d887978d0e57fb13b5f96d'),
}


HISTORY_PROFILE_POLICY = 'native-keep-history-profile-metadata-v1'
# Exact independently reviewed e87 profile overlay on both qualified runtimes.
# The shared capacity identity normalizer is the sole additional module.
HISTORY_PROFILE_COMPOSITIONS = {
    'lfm': {
        'data/port_record_graph.py': (
            'e756edd8c6b8b61f51da78527145ee69b680655d6ad067c03a696953967f6cef',
            '51db892a7c43bcaef1661f856da4d61890324072c699560c37c26e4191bc5672'),
        'data/text_corpus.py': (
            '004659afb57d4c23dd64262ed49ca4b3964938b60e701789a61b63091464773a',
            '622b4ad08257ed09d0dd10d52d97adfa1c45cc8b9c8c46bbeef57fae6caff63b'),
        'model/native_bundle.py': (
            '579e0e1010f5e5ba96950c03ab56c7abf5afe5bd93880bdab4608fcd0bcfb33d',
            '64ac96039b8111c25909f86fe2f19692ed0bdb0288e84f0812b5f5193ab63773'),
        'model/native_composite.py': (
            '4bd042dc5d6572dabe76ff8022b98f4cf5e680aa1120251496292a9ad06d4c1c',
            '59a5baa6eb56c5f79e77abfc7321b812c82ecfa855374fa030516944fc34554e'),
        'model/native_model.py': (
            'd9dedf39ca629c758c0351df73d5b6ad85bb868a5facff673f3cba7a09bfb4d4',
            '4345f99f74d37d4a2b184a7df741b9e735b6d9e395786ee92a2970b09307f419'),
        'model/semantic_channel.py': (
            'a8c75246738cac7ce8b448743d7e43d64c05c2ed132c5c44d47aa0f8f267303b',
            'fe501e9eb3580af4a8390e1a3b1dd826b101d9af9578f655a80cd6d2ce91211d'),
        'rl/native_raw_policy.py': (
            '77676a8adbb280e0c905912477479d0ec5d1211c12e8f072c1b5a9f405061668',
            '585544d610fffa0e1dc275d1a56f9d331989a58383639835550055cbe77e2a32'),
        'rl/native_stage.py': (
            '9e8559a9e96c4ed94194a6b3bf0c9803b308d475380e0485f9b9c6e2c3997f8f',
            '46c1e117ba73a86e9ad2e5fcfab306ba4ac3005945cf794a4304f5392ecf7c77'),
        'rl/production.py': (
            'd5e8a7b871f43d4e05282b9bdec399fbebde356d419b5066c82600c5ebec203b',
            '2639c62caf1035a39b39c7081145a69e2152a90449b3246fa8023aadb5ab3acc'),
        'serve/__init__.py': (
            '5c32ecdf1bc053acbd99e8cd1409eff07ddc5b6fc4ca403a5f12f660fcda1571',
            'bb1eda9824538f2037d2cdb23e162cc043bca70ed06dcb2a8e3ff80b74f0f10d'),
        'serve/chat.py': (
            '12e9dfca075a1159b72fc5f2be1b7b376c65e5e8780a766adc62ad78a4ceea36',
            '5a7e076684f3a0b1f590fac2ee3b6df98facf9999af67b34ee09d3f6103c22a0'),
        'serve/engine.py': (
            'b359949d90172b40152cef668a9abc44ee20a53475d29a43c964e669f4aa1ef1',
            '2dfc0c6070085e29a794ea4a82972a6978947b501555eacede13e33e79dce0cd'),
        'train/native_continuation.py': (
            'facd6a6f8f6e68799cd059d2b2ac96cc3679211b9d541fcfaa3274837f186f71',
            '10335621edfd58fe400818daad221e1fca7dd2b843305cdb6c8ac4422e8f01c3'),
        'train/native_semantic_stage.py': (
            '66462611c9d41e39bf403802668efdf21a75c49953e143b6acdac5efbc44e986',
            '8d456cc510b723aa1d940c285bd4b978415eff49096b9007759f3a4439e29801'),
        'train/native_state_sft.py': (
            '45c5c8da86060ddb431bdcb24de97daefcc19fecac9398fbff399399d8dd18ba',
            '2380a03c921d1ef840f1ba80cffdd3af69b61237af9d5c912782495946943e71'),
        'train/recipe.py': (
            '32b3c41456a24683b1f4024eaefe4c59c2fd96d49e24d18615f1eb0a59fa62c8',
            '9e1538a56d97c32c517a686ec81616a8acbae570d1825a1e66d6fbd671116e14'),
        'train/native_resident.py': (
            '32048a86963cc970a9a9c04b4292d33be4357810ce7f50dfc2191e17db9822dd',
            '441bceb6c65f0654ab5bbc2c4f4b97f7d4ffbd404029488ff533c3b76b1442cb'),
    },
    'mellum': {
        'data/port_record_graph.py': (
            'e756edd8c6b8b61f51da78527145ee69b680655d6ad067c03a696953967f6cef',
            '51db892a7c43bcaef1661f856da4d61890324072c699560c37c26e4191bc5672'),
        'data/text_corpus.py': (
            '004659afb57d4c23dd64262ed49ca4b3964938b60e701789a61b63091464773a',
            '622b4ad08257ed09d0dd10d52d97adfa1c45cc8b9c8c46bbeef57fae6caff63b'),
        'model/native_bundle.py': (
            '579e0e1010f5e5ba96950c03ab56c7abf5afe5bd93880bdab4608fcd0bcfb33d',
            '64ac96039b8111c25909f86fe2f19692ed0bdb0288e84f0812b5f5193ab63773'),
        'model/native_composite.py': (
            '4bd042dc5d6572dabe76ff8022b98f4cf5e680aa1120251496292a9ad06d4c1c',
            '59a5baa6eb56c5f79e77abfc7321b812c82ecfa855374fa030516944fc34554e'),
        'model/native_model.py': (
            'd9dedf39ca629c758c0351df73d5b6ad85bb868a5facff673f3cba7a09bfb4d4',
            '4345f99f74d37d4a2b184a7df741b9e735b6d9e395786ee92a2970b09307f419'),
        'model/semantic_channel.py': (
            'a8c75246738cac7ce8b448743d7e43d64c05c2ed132c5c44d47aa0f8f267303b',
            'fe501e9eb3580af4a8390e1a3b1dd826b101d9af9578f655a80cd6d2ce91211d'),
        'rl/native_raw_policy.py': (
            '77676a8adbb280e0c905912477479d0ec5d1211c12e8f072c1b5a9f405061668',
            '585544d610fffa0e1dc275d1a56f9d331989a58383639835550055cbe77e2a32'),
        'rl/native_stage.py': (
            '9e8559a9e96c4ed94194a6b3bf0c9803b308d475380e0485f9b9c6e2c3997f8f',
            '46c1e117ba73a86e9ad2e5fcfab306ba4ac3005945cf794a4304f5392ecf7c77'),
        'rl/production.py': (
            'd5e8a7b871f43d4e05282b9bdec399fbebde356d419b5066c82600c5ebec203b',
            '2639c62caf1035a39b39c7081145a69e2152a90449b3246fa8023aadb5ab3acc'),
        'serve/__init__.py': (
            '5c32ecdf1bc053acbd99e8cd1409eff07ddc5b6fc4ca403a5f12f660fcda1571',
            'bb1eda9824538f2037d2cdb23e162cc043bca70ed06dcb2a8e3ff80b74f0f10d'),
        'serve/chat.py': (
            '12e9dfca075a1159b72fc5f2be1b7b376c65e5e8780a766adc62ad78a4ceea36',
            '5a7e076684f3a0b1f590fac2ee3b6df98facf9999af67b34ee09d3f6103c22a0'),
        'serve/engine.py': (
            'b359949d90172b40152cef668a9abc44ee20a53475d29a43c964e669f4aa1ef1',
            '2dfc0c6070085e29a794ea4a82972a6978947b501555eacede13e33e79dce0cd'),
        'train/native_continuation.py': (
            'facd6a6f8f6e68799cd059d2b2ac96cc3679211b9d541fcfaa3274837f186f71',
            '10335621edfd58fe400818daad221e1fca7dd2b843305cdb6c8ac4422e8f01c3'),
        'train/native_semantic_stage.py': (
            '66462611c9d41e39bf403802668efdf21a75c49953e143b6acdac5efbc44e986',
            '8d456cc510b723aa1d940c285bd4b978415eff49096b9007759f3a4439e29801'),
        'train/native_state_sft.py': (
            '77f51bb6de67f396a5e25f96a8ed9b4e7e4c9ea96ba3c04e7e6fb5492e4ae6fe',
            '3b17ff15ee3f6a8ed67992fa4b090659208b59ce01eefaee426b6ea537a28c17'),
        'train/recipe.py': (
            '32b3c41456a24683b1f4024eaefe4c59c2fd96d49e24d18615f1eb0a59fa62c8',
            '9e1538a56d97c32c517a686ec81616a8acbae570d1825a1e66d6fbd671116e14'),
        'train/native_resident.py': (
            '952926630260bb95187afaaf442292a216bf21089e379db28980a244723b14ff',
            '441bceb6c65f0654ab5bbc2c4f4b97f7d4ffbd404029488ff533c3b76b1442cb'),
    },
}
HISTORY_PROFILE_SOURCE_POLICY_FILES = ('data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py', 'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')


HISTORY_PROFILE_VERIFIER = 'train/native_capacity_continuation.py'


def _history_profile_verifier(review, *, executing=False):
    old,new=review['old_code'],review['new_code']
    if old.get(HISTORY_PROFILE_VERIFIER)==new.get(HISTORY_PROFILE_VERIFIER):return
    witness=review.get('verifier_bootstrap') or {}
    if (not re.fullmatch(r'[0-9a-f]{40}',witness.get('source_revision',''))
            or not old.get(HISTORY_PROFILE_VERIFIER) or not new.get(HISTORY_PROFILE_VERIFIER)):
        raise ValueError('KEEP verifier bootstrap needs an immutable published source reference')
    for key,expected in (('old',old[HISTORY_PROFILE_VERIFIER]),
                         ('new',new[HISTORY_PROFILE_VERIFIER]),
                         ('source',new[HISTORY_PROFILE_VERIFIER])):
        reference=witness.get(key) or {}
        if reference.get('sha256')!=expected:
            raise ValueError('KEEP verifier bootstrap differs from approved inventory')
        bound(reference)
    # Only generation binds the executing helper. Historical verification uses
    # its immutable source reference, so later verifier upgrades retain ancestry.
    if executing and binding(Path(__file__))['sha256']!=new[HISTORY_PROFILE_VERIFIER]:
        raise ValueError('KEEP verifier bootstrap is not the executing helper')


def _history_profile_contract(review):
    if review['optimizer_policy'] != HISTORY_PROFILE_POLICY:return
    config=(review.get('identity') or {}).get('model_config') or {}
    old,new=review['old_code'],review['new_code']
    changes={name for name in old.keys() | new.keys() if old.get(name)!=new.get(name)}
    if ((review.get('identity') or {}).get('objective') is not None
            or config.get('history_reasoning_policy') != 'keep'
            or review.get('fresh_policy_capacity_required') is not True
            or not changes
            or not any(changes <= set(composition) | {HISTORY_PROFILE_VERIFIER} and all(
                old.get(name) in pair and new.get(name)==pair[1]
                for name,pair in composition.items())
                for composition in HISTORY_PROFILE_COMPOSITIONS.values())
            or any(not old.get(name) or old.get(name)!=new.get(name)
                   for name in HISTORY_PROFILE_SOURCE_POLICY_FILES)):
        raise ValueError('KEEP profile continuation needs exact reviewed module pairs and unchanged source parser; policy capacity stays fresh')
    _history_profile_verifier(review)


def allowed_changes(policy):
    return {POLICY: ALLOWED_CHANGES, METADATA_POLICY: METADATA_CHANGES,
            HISTORY_PROFILE_POLICY: set(next(iter(HISTORY_PROFILE_COMPOSITIONS.values()))) | {HISTORY_PROFILE_VERIFIER},
            LOADER_IO_POLICY: LOADER_IO_CHANGES,
            LINEAGE_METADATA_POLICY: {LINEAGE_METADATA_FILE},
            POLICY_GATE_SFT_POLICY: set(POLICY_GATE_SFT_COMPOSITION),
            ORDINARY_WRITER_POLICY: set(ORDINARY_WRITER_COMPOSITION),
            WHOLE_HEAD_SOURCE_POLICY: set(WHOLE_HEAD_SOURCE_CHANGES),
            ATTENTION_INDEX_POLICY: {'model/shared_state.py'}}.get(policy, set())


def _lineage_metadata_contract(review):
    if review['optimizer_policy'] != LINEAGE_METADATA_POLICY:
        return
    actual = (review['old_code'].get(LINEAGE_METADATA_FILE),
              review['new_code'].get(LINEAGE_METADATA_FILE))
    if actual != LINEAGE_METADATA_COMPOSITION:
        raise ValueError('saved recipe metadata continuation requires the exact independently reviewed trainer repair')


def _attention_index_contract(review, limits):
    if review['optimizer_policy'] != ATTENTION_INDEX_POLICY:
        return
    basis = review.get('attention_index_basis') or {}
    composition = ATTENTION_INDEX_COMPOSITIONS.get(basis.get('model'))
    reference = basis.get('review') or {}
    if composition is None or reference.get('sha256') != ATTENTION_INDEX_REVIEW_SHA256:
        raise ValueError('attention index continuation requires its exact independently approved frozen review')
    approved = json.loads(bound(reference).read_text())
    parent, candidate, old_sha, new_sha = composition
    report = next((row for row in approved.get('reports', []) if row.get('model') == basis['model']), {})
    if (basis.get('parent_commit') != parent or basis.get('candidate_commit') != candidate
            or report.get('parent') != parent or report.get('candidate') != candidate
            or report.get('shared_state_sha256') != new_sha
            or review['old_code'].get('model/shared_state.py') != old_sha
            or review['new_code'].get('model/shared_state.py') != new_sha):
        raise ValueError('attention index continuation changed the exact reviewed composition')
    regressions = json.loads(bound(basis.get('cuda_regressions')).read_text())
    expected = {(values[1], port) for values in ATTENTION_INDEX_COMPOSITIONS.values()
                for port in ('lfm', 'mellum')}
    cases = regressions.get('cases') or []
    if (regressions.get('schema') != 'natlang.attention-index-cuda-regressions/1'
            or len(cases) != 4 or {(row.get('candidate_commit'), row.get('port')) for row in cases} != expected):
        raise ValueError('attention index continuation requires all four frozen both-port CUDA cases')
    for case in cases:
        if any(case.get(key) is not True for key in ('passed', 'cuda', 'outputs_bitwise',
                'all_named_adjoints_bitwise', 'checkpoint_recompute', 'distinct_graph_lifetime')):
            raise ValueError('attention index CUDA regression is not strict bitwise qualification')
        bound(case.get('log'))
    paired = json.loads(bound(basis.get('normal_batch_paired')).read_text())
    normal = paired.get('batch') or {}
    if (paired.get('schema') != 'natlang.attention-index-production-paired/1'
            or paired.get('passed') is not True or not same_geometry_identity(paired.get('identity'), review['identity'])
            or paired.get('old_runtime_code_sha256') != digest(review['old_code'])
            or paired.get('new_runtime_code_sha256') != digest(review['new_code'])
            or paired.get('parent_checkpoint_sha256') != basis.get('normal_probe_parent_checkpoint_sha256')
            or not isinstance(paired.get('parent_checkpoint_sha256'), str)
            or len(paired['parent_checkpoint_sha256']) != 64
            or paired.get('optimizer_updates') != 0
            or any(paired.get(key) is not True for key in ('cuda_synchronized', 'outputs_bitwise',
                'loss_bitwise', 'all_named_adjoints_bitwise', 'optimizer_state_unchanged',
                'rng_restored', 'source_credits_unchanged'))):
        raise ValueError('attention index continuation lacks an exact unchanged-state production paired result')
    for key in ('forward_tokens', 'kv_slots'):
        if type(normal.get(key)) is not int or not 0 < normal[key] <= limits[key]:
            raise ValueError('attention index paired batch differs from the inherited geometry')
    if normal != basis.get('normal_batch_geometry'):
        raise ValueError('attention index paired batch differs from the independently reviewed normal batch')
    bound(paired.get('source'))
    for key in ('old_seconds', 'new_seconds'):
        value = paired.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('attention index paired timings require actual synchronized positive durations')
    memory = basis.get('index_memory') or {}
    old_count = memory.get('old_attention_index_allocations')
    index_bytes = limits['forward_tokens'] * 8
    extra = review.get('metadata_extra_bytes')
    if (type(old_count) is not int or old_count < 1
            or memory.get('new_attention_index_allocations') != 1
            or memory.get('index_dtype') != 'torch.int64'
            or memory.get('max_index_elements') != limits['forward_tokens']
            or memory.get('new_index_bytes') != index_bytes
            or memory.get('old_index_bytes') != old_count * index_bytes
            or type(memory.get('host_workspace_extra_bytes_upper_bound')) is not int
            or type(extra) is not int or extra < (1 << 30)
            or not 0 <= memory['host_workspace_extra_bytes_upper_bound'] <= extra):
        raise ValueError('attention index continuation requires conservative explicit index/workspace memory bounds')


def _memory_allowance(review):
    parameters = review.get('ordinary_parameters') or []
    if not parameters or len({p['name'] for p in parameters}) != len(parameters):
        raise ValueError('capacity continuation requires distinct actual ordinary parameters')
    for parameter in parameters:
        if (parameter.get('dtype') != 'torch.float32' or not parameter.get('shape')
                or any(type(n) is not int or n < 1 for n in parameter['shape'])):
            raise ValueError('capacity continuation requires the unchanged FP32 ordinary layout')
    if review['optimizer_policy'] in {METADATA_POLICY, LOADER_IO_POLICY, ATTENTION_INDEX_POLICY, LINEAGE_METADATA_POLICY, POLICY_GATE_SFT_POLICY, ORDINARY_WRITER_POLICY, WHOLE_HEAD_SOURCE_POLICY, HISTORY_PROFILE_POLICY}:
        extra = review.get('metadata_extra_bytes')
        if (review.get('optimizer_state_unchanged') is not True
                or type(extra) is not int or extra < (64 << 20)):
            raise ValueError('metadata continuation requires unchanged optimizer state and >=64MiB explicit working-memory allowance')
        return extra
    return 64 * sum(math.prod(p['shape']) for p in parameters)


def _proof_cases(reference, *, identity, code_sha, parent, seen=()):
    value = json.loads(bound(reference).read_text())
    if reference['sha256'] in seen:
        raise ValueError('capacity continuation evidence contains a cycle')
    seen = (*seen, reference['sha256'])
    if value.get('schema') == 'natlang.native-capacity-continuation/1':
        review = json.loads(bound(value['review']).read_text())
        certificate = json.loads(bound(value['source_certificate']).read_text())
        changes = {name: {'old': review['old_code'].get(name), 'new': review['new_code'].get(name)}
            for name in review['old_code'].keys() | review['new_code'].keys()
            if review['old_code'].get(name) != review['new_code'].get(name)}
        policy = review.get('optimizer_policy')
        allowed = allowed_changes(policy)
        if (value.get('passed') is not True or not same_geometry_identity(value.get('identity'), identity)
                or value.get('runtime_code_sha256') != code_sha or value.get('optimizer_updates') != 0
                or value.get('new_parent_measured') is not False
                or value.get('continued_parent_checkpoint_sha256') != parent
                or review.get('schema') != 'natlang.native-capacity-continuation-review/1'
                or review.get('approved') is not True or review.get('parameter_layout_unchanged') is not True
                or value.get('continuation_policy', policy) != policy
                or not changes or set(changes) - allowed or changes != review.get('changes')
                or digest(review['new_code']) != code_sha
                or digest(review['old_code']) != value.get('old_runtime_code_sha256')
                or not same_geometry_identity(review.get('identity'), identity)
                or review.get('source_certificate') != value['source_certificate']
                or review.get('new_parent_checkpoint_sha256') != parent
                or review.get('old_parent_checkpoint_sha256') != value.get('source_parent_checkpoint_sha256', value.get('measured_parent_checkpoint_sha256'))
                or certificate.get('passed') is not True or not same_geometry_identity(certificate.get('identity'), identity)
                or certificate.get('schema') != 'natlang.native-resident-geometry-qualification/1'
                or certificate.get('runtime_code_sha256') != value['old_runtime_code_sha256']
                or certificate.get('evidence') != value.get('source_evidence')
                or certificate.get('limits') != value.get('limits')):
            raise ValueError('inherited capacity continuation differs from its approved evidence chain')
        cases = _proof_cases(value['source_evidence'], identity=identity,
            code_sha=value['old_runtime_code_sha256'],
            parent=value.get('source_parent_checkpoint_sha256', value['measured_parent_checkpoint_sha256']), seen=seen)
        origin = json.loads(bound(value['source_evidence']).read_text()).get(
            'measured_parent_checkpoint_sha256', review['old_parent_checkpoint_sha256'])
        if value['measured_parent_checkpoint_sha256'] != origin:
            raise ValueError('inherited capacity changed its actual measured origin')
        if {key: max(case[key] for case in cases) for key in ('forward_tokens', 'kv_slots')} != value['limits']:
            raise ValueError('inherited capacity limits differ from actual source cases')
        _attention_index_contract(review, value['limits'])
        _lineage_metadata_contract(review)
        _policy_gate_sft_contract(review)
        _ordinary_writer_contract(review)
        _whole_head_source_contract(review)
        _history_profile_contract(review)
        extra = value.get('extra_bytes_per_memory_domain')
        if type(extra) is not int or extra != _memory_allowance(review):
            raise ValueError('inherited capacity lost its additional memory allowance')
        return [{**case, 'inherited_extra_bytes': case.get('inherited_extra_bytes', 0) + extra} for case in cases]
    if (value.get('passed') is not True or not same_geometry_identity(value.get('identity'), identity)
            or value.get('runtime_code_sha256') != code_sha
            or value.get('parent_checkpoint_sha256') != parent
            or value.get('optimizer_updates') != 0):
        raise ValueError('capacity source proof differs from its measured parent/runtime')
    if value.get('schema') == 'natlang.native-memory-combined-capacity/1':
        children = value.get('proofs') or []
        cases = [case for child in children for case in _proof_cases(child,
            identity=identity, code_sha=code_sha, parent=parent, seen=seen)]
        if not cases or cases != (value.get('capacity') or {}).get('cases'):
            raise ValueError('combined source cases differ from their actual bound proofs')
        return cases
    if (value.get('schema') != 'natlang.native-memory-qualification/1'
            or (value.get('numerical_reference') or {}).get('passed') is not True
            or (value.get('capacity') or {}).get('passed') is not True):
        raise ValueError('continuation requires actual numerical and capacity evidence')
    cases = value['capacity'].get('cases') or []
    if not cases or any(case.get('status') != 'passed' for case in cases):
        raise ValueError('continuation requires successful actual whole-graph cases')
    if any('inherited_extra_bytes' in case for case in cases):
        raise ValueError('raw measured cases cannot declare inherited memory allowances')
    return cases


def verify_capacity_certificate(reference, *, identity, code, parent):
    """Verify the complete existing evidence chain for its actual start parent."""
    from .native_resident import geometry_capacity
    limits = geometry_capacity(reference, identity=identity, code=code)
    certificate = json.loads(bound(reference).read_text())
    cases = _proof_cases(certificate['evidence'], identity=identity,
                         code_sha=digest(code), parent=parent)
    if {key: max(case[key] for case in cases) for key in ('forward_tokens', 'kv_slots')} != limits:
        raise ValueError('capacity certificate limits differ from its complete measured cases')
    return limits


def capacity_certificate_dependencies(reference, *, identity, code, parent):
    """Record exactly the metadata consumed by the existing strict verifier."""
    files={};token=_DEPENDENCY_SINK.set(files)
    try:verify_capacity_certificate(reference,identity=identity,code=code,parent=parent)
    finally:_DEPENDENCY_SINK.reset(token)
    return files


def continue_capacity(review_binding, *, old_package, new_package, out):
    review = json.loads(bound(review_binding).read_text())
    if (review.get('schema') != 'natlang.native-capacity-continuation-review/1'
            or review.get('approved') is not True
            or review.get('parameter_layout_unchanged') is not True
            or review.get('optimizer_policy') not in {POLICY, METADATA_POLICY, LOADER_IO_POLICY, ATTENTION_INDEX_POLICY, LINEAGE_METADATA_POLICY, POLICY_GATE_SFT_POLICY, ORDINARY_WRITER_POLICY, WHOLE_HEAD_SOURCE_POLICY, HISTORY_PROFILE_POLICY}):
        raise ValueError('capacity continuation requires an approved exact optimizer review')
    old, new = code_inventory(old_package), code_inventory(new_package)
    if old != review.get('old_code') or new != review.get('new_code'):
        raise ValueError('capacity continuation runtime inventory differs from reviewed bytes')
    changes = {name: {'old': old.get(name), 'new': new.get(name)}
        for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    policy = review['optimizer_policy']
    allowed = allowed_changes(policy)
    if not changes or set(changes) - allowed or changes != review.get('changes'):
        raise ValueError('capacity continuation contains unreviewed forward/runtime changes')
    identity = review['identity']
    source = json.loads(bound(review['source_certificate']).read_text())
    if (source.get('schema') != 'natlang.native-resident-geometry-qualification/1'
            or source.get('passed') is not True or not same_geometry_identity(source.get('identity'), identity)
            or source.get('runtime_code_sha256') != digest(old)):
        raise ValueError('capacity identity changed: model/packing/offload/backend must match')
    cases = _proof_cases(source['evidence'], identity=identity, code_sha=digest(old),
        parent=review['old_parent_checkpoint_sha256'])
    limits = {key: max(case[key] for case in cases) for key in ('forward_tokens', 'kv_slots')}
    if (source.get('limits') != limits or any(type(v) is not int or v < 1 for v in limits.values())):
        raise ValueError('capacity limits differ from actual source cases')
    _attention_index_contract(review, limits)
    _lineage_metadata_contract(review)
    _policy_gate_sft_contract(review)
    _ordinary_writer_contract(review)
    _whole_head_source_contract(review)
    _history_profile_contract(review)
    if policy == HISTORY_PROFILE_POLICY:_history_profile_verifier(review, executing=True)
    # Adam first/second moments, retained old buffers, telemetry snapshots and
    # per-parameter update intermediates. No backbone-sized copy is authorized.
    extra = _memory_allowance(review)
    reserve = review.get('reserve_bytes')
    device_limit, host_limit = review.get('device_memory_bytes'), review.get('host_memory_limit_bytes')
    if (type(reserve) is not int or reserve < (1 << 30)
            or any(type(n) is not int or n <= 0 for n in (device_limit, host_limit))
            or identity.get('runtime', {}).get('total_memory_bytes') != device_limit):
        raise ValueError('capacity continuation requires actual hardware limits and >=1GiB reserve')
    peaks = []
    for case in cases:
        device_peak = case.get('cuda_peak_allocated_bytes')
        host_peak = (case.get('host_peak') or {}).get('cgroup_current_bytes')
        if any(type(n) is not int or n <= 0 for n in (device_peak, host_peak)):
            raise ValueError('capacity continuation requires measured device and cgroup peaks')
        inherited_extra = case.get('inherited_extra_bytes', 0)
        if device_peak + inherited_extra + extra + reserve >= device_limit or host_peak + inherited_extra + extra + reserve >= host_limit:
            raise ValueError('capacity continuation exceeds measured memory headroom')
        peaks.append({'id': case['id'], 'device_bytes': device_peak, 'host_bytes': host_peak,
                      **({'inherited_extra_bytes': inherited_extra} if inherited_extra else {})})
    result = {'schema': 'natlang.native-capacity-continuation/1', 'passed': True,
        'review': review_binding, 'source_certificate': review['source_certificate'],
        'source_evidence': source['evidence'], 'identity': identity,
        'runtime_code_sha256': digest(new), 'old_runtime_code_sha256': digest(old),
        'measured_parent_checkpoint_sha256': json.loads(bound(source['evidence']).read_text()).get(
            'measured_parent_checkpoint_sha256', review['old_parent_checkpoint_sha256']),
        'source_parent_checkpoint_sha256': review['old_parent_checkpoint_sha256'],
        'continued_parent_checkpoint_sha256': review['new_parent_checkpoint_sha256'],
        'new_parent_measured': False, 'extra_bytes_per_memory_domain': extra,
        'continuation_policy': policy,
        'reserve_bytes': reserve, 'measured_peaks': peaks, 'optimizer_updates': 0,
        'task_quality_qualified': False, 'limits': limits}
    directory = Path(out)/digest(result)
    directory.mkdir(parents=True, exist_ok=True)
    evidence, certificate = directory/'continuation.json', directory/'capacity.json'
    def immutable(path, value):
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('immutable capacity continuation changed')
        else:
            write_json_durable(path, value)
    immutable(evidence, result)
    immutable(certificate, {'schema': 'natlang.native-resident-geometry-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(new),
        'limits': limits, 'evidence': binding(evidence),
        'scope': 'inherited actual capacity through explicitly reviewed ' +
            ('optimizer/telemetry-only' if policy == POLICY else
             'exact frozen attention-index-only' if policy == ATTENTION_INDEX_POLICY else
             'exact frozen saved-recipe-metadata-only' if policy == LINEAGE_METADATA_POLICY else
             'exact frozen policy-gate ordinary-SFT-only (fresh policy measurement required)' if policy == POLICY_GATE_SFT_POLICY else
             'exact frozen ordinary-SFT report-writer-only (fresh policy measurement required)' if policy == ORDINARY_WRITER_POLICY else
             'whole shared HEAD source-v1 ordinary-SFT-only (fresh policy measurement required)' if policy == WHOLE_HEAD_SOURCE_POLICY else
             'CPU scheduling/pending-obligation-only') +
            ' continuation; new parent not measured'})
    return certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, help='JSON exact path/SHA binding to independently approved review')
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(binding(continue_capacity(json.loads(args.review),
        old_package=args.old_package, new_package=args.new_package, out=args.out))))


if __name__ == '__main__':
    main()
