import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

from build_native_turn_union import plan_member_rows, _row_identity  # noqa: E402
from source_review import pending_unpinned_source_groups  # noqa: E402
import target_set_manifest  # noqa: E402


TOKENS = ['lfm', 'mellum']


def test_union_compression_preserves_exact_utf8_and_decoded_receipt(tmp_path):
    from build_native_turn_union import union_text_writer
    from natlang_neuralese.common.jsonio import open_jsonl_text
    text = '{"text":"café 😀"}\r\n{"text":"next"}\n'
    plain = tmp_path / 'text-data.jsonl'
    packed = tmp_path / 'text-data.jsonl.zst'
    with union_text_writer(plain) as ordinary:
        ordinary.write(text)
    with union_text_writer(packed, True) as compressed:
        compressed.write(text[:12])
        compressed.write(text[12:])
    with open_jsonl_text(packed) as stream:
        restored = stream.read().encode('utf-8')
    assert restored == plain.read_bytes() == text.encode('utf-8')
    assert compressed.decoded_sha256 == hashlib.sha256(restored).hexdigest()
    assert compressed.decoded_bytes == len(restored)


def test_reuse_verifies_plain_and_compressed_files_without_rewriting(tmp_path):
    from build_native_turn_union import union_text_writer
    lines = ['{"text":"café 😀"}\r\n', '{"text":"next"}\n']
    for compressed in (False, True):
        path = tmp_path / ('text.jsonl.zst' if compressed else 'text.jsonl')
        with union_text_writer(path, compressed) as writer:
            for line in lines:
                writer.write(line)
        before = path.stat()
        original = path.read_bytes()
        with union_text_writer(path, compressed, reuse_existing=True) as reused:
            for line in lines:
                reused.write(line)
        after = path.stat()
        assert path.read_bytes() == original
        assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
        assert reused.decoded_sha256 == writer.decoded_sha256
        assert reused.decoded_bytes == writer.decoded_bytes


def test_reuse_rejects_changed_selection_and_extra_rows_without_mutating_file(tmp_path):
    import pytest
    from build_native_turn_union import union_text_writer
    path = tmp_path / 'text.jsonl.zst'
    with union_text_writer(path, True) as writer:
        writer.write('{"id":"one"}\n')
        writer.write('{"id":"two"}\n')
    original = path.read_bytes()
    for selected in [['{"id":"different"}\n'], ['{"id":"one"}\n'],
                     ['{"id":"one"}\n', '{"id":"two"}\n', '{"id":"three"}\n']]:
        with pytest.raises(ValueError, match='existing union'):
            with union_text_writer(path, True, reuse_existing=True) as writer:
                for line in selected:
                    writer.write(line)
        assert path.read_bytes() == original


def test_removed_quality_reference_blocks_derivatives_in_both_splits():
    candidates = {
        'base': {tok: {'train-copy': identity('train', groups=('question-held',)),
                       'test-copy': identity('test', split='test', groups=('question-held',)),
                       'good': identity('good', groups=('independent',))}
                 for tok in TOKENS}}
    selected, omitted = plan_member_rows(['base'], candidates, {}, TOKENS, set(), set(),
                                        quarantined_source_groups={'question-held'})
    assert selected == {('base', 'good')}
    assert {o['reason'] for o in omitted.values()} == {'source_quality_quarantine'}


def test_exact_decision_hold_drops_only_its_decision_and_containing_chain():
    candidates = {'base': {tok: {
        'bad-decision': {**identity('bad'), 'source_record_ids': ('decision:bad',)},
        'containing-chain': {**identity('chain'), 'source_record_ids': ('decision:good', 'decision:bad')},
        'good-decision': {**identity('good'), 'source_record_ids': ('decision:good',)},
    } for tok in TOKENS}}
    selected, omitted = plan_member_rows(['base'], candidates, {}, TOKENS, set(), set(),
                                         held_decision_ids={'decision:bad'})
    assert selected == {('base', 'good-decision')}
    assert {key: item['reason'] for key, item in omitted.items()} == {
        ('base', 'bad-decision'): 'held_native_decision',
        ('base', 'containing-chain'): 'held_native_decision',
    }


def test_removed_member_held_group_still_blocks_a_train_derivative():
    candidates = {'remaining': {tok: {'r': identity('new text', groups=('removed-held-group',))}
                                for tok in TOKENS}}
    selected, omitted = plan_member_rows(['remaining'], candidates, {}, TOKENS,
                                        {'removed-held-group'}, set())
    assert not selected
    assert omitted[('remaining', 'r')]['reason'] == 'train_document_shares_held_source_group'


def identity(text, split='train', groups=('g1',)):
    return {
        'split': split,
        'source_groups': tuple(sorted(groups)),
        'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }


def member(name, rows_by_tokenizer):
    return {name: {tok: dict(rows) for tok, rows in rows_by_tokenizer.items()}}


def decide(order, members, *, supersedes=None, held_groups=None, held_texts=None, duplicates=None,
           source_review_groups=None):
    return plan_member_rows(order, members, supersedes or {}, TOKENS,
                            held_groups or set(), held_texts or set(), member_duplicates=duplicates,
                            source_review_groups=source_review_groups)


def test_pending_unpinned_source_reviews_drop_train_twins_before_dedup(tmp_path):
    registry = tmp_path / 'source-reviews'
    registry.mkdir()
    (registry / 'holds.jsonl').write_text(
        '{"dataset":"musique","id":"bad","status":"pending"}\n'
        '{"dataset":"musique","id":"pinned","status":"pending","sourceSnapshotSha256":"abc"}\n'
        '{"dataset":"musique","id":"clear","status":"resolved"}\n')
    held = pending_unpinned_source_groups(registry)
    assert held == {'musique:bad'}
    reviewed = member('reviewed', {tok: {'doc': identity('same', groups=('musique:bad',))} for tok in TOKENS})
    eligible = member('eligible', {tok: {'doc': identity('new', groups=('musique:other',))} for tok in TOKENS})
    selected, omissions = decide(['reviewed', 'eligible'], reviewed | eligible, source_review_groups=held)
    assert selected == {('eligible', 'doc')}
    assert omissions[('reviewed', 'doc')]['reason'] == 'source_review_pending'
    assert omissions[('reviewed', 'doc')]['source_groups'] == ['musique:bad']
    pinned = member('pinned', {tok: {'p': identity('pinned', groups=('musique:pinned',))} for tok in TOKENS})
    selected, omissions = decide(['pinned'], pinned, source_review_groups=held)
    assert selected == {('pinned', 'p')}
    assert not omissions


def test_incomplete_twin_cannot_claim_same_document_id_or_text():
    older = member('older', {'lfm': {'doc': identity('same')}})
    newer = member('newer', {'lfm': {'doc': identity('same')}, 'mellum': {'doc': identity('same')}})
    selected, omissions = decide(['older', 'newer'], older | newer)
    assert selected == {('newer', 'doc')}
    assert omissions[('older', 'doc')]['reason'] == 'missing_tokenizer_twin'


def test_mismatched_tokenizer_twin_identity_cannot_claim_document_id():
    mismatched = member('mismatch', {'lfm': {'doc': identity('left')},
                                     'mellum': {'doc': identity('right')}})
    valid = member('valid', {tok: {'doc': identity('same')} for tok in TOKENS})
    selected, omissions = decide(['mismatch', 'valid'], mismatched | valid)
    assert selected == {('valid', 'doc')}
    assert omissions[('mismatch', 'doc')]['reason'] == 'tokenizer_twin_identity_mismatch'


def native_identity(text, source='source:doc', **extra):
    return _row_identity({'id': 'doc', 'split': 'train', 'source_groups': ['g1'],
                          'source_record_ids': [source], 'text': text, **extra})


def test_backbone_specific_chat_templates_preserve_one_logical_document():
    paired = member('native', {'lfm': {'doc': native_identity('<|startoftext|><|im_start|>user\nhello')},
                               'mellum': {'doc': native_identity('<|im_start|>user\nhello')}})
    selected, omissions = decide(['native'], paired)
    assert selected == {('native', 'doc')}
    assert not omissions


def test_source_or_context_mismatch_still_rejects_a_tokenizer_twin():
    left = native_identity('left template')
    for right in (native_identity('right template', source='other-source'),
                  native_identity('right template', neuralese_context_attestations=[{'sha256': 'changed'}]),
                  native_identity('right template', source_records=[{'id': 'source:doc', 'sha256': 'changed'}])):
        selected, omissions = decide(['native'], member('native', {'lfm': {'doc': left},
                                                                  'mellum': {'doc': right}}))
        assert not selected
        assert omissions[('native', 'doc')]['reason'] == 'tokenizer_twin_identity_mismatch'


def test_exact_text_dedup_requires_both_backbone_renderings_to_match():
    def pair(name, rid, lfm, mellum):
        return member(name, {tok: {rid: _row_identity({'id': rid, 'split': 'train', 'source_groups': ['g1'],
                                                       'source_record_ids': [rid], 'text': text})}
                             for tok, text in [('lfm', lfm), ('mellum', mellum)]})
    first = pair('first', 'a', 'same lfm', 'mellum a')
    second = pair('second', 'b', 'same lfm', 'mellum b')
    selected, _ = decide(['first', 'second'], first | second)
    assert selected == {('first', 'a'), ('second', 'b')}


def test_held_or_decontaminated_copy_does_not_claim_document_id():
    held_copy = member('held-copy', {tok: {'doc': identity('held', groups=('reserved',))} for tok in TOKENS})
    eligible_copy = member('eligible-copy', {tok: {'doc': identity('new', groups=('train',))} for tok in TOKENS})
    selected, omissions = decide(['held-copy', 'eligible-copy'], held_copy | eligible_copy,
                                 held_groups={'reserved'})
    assert selected == {('eligible-copy', 'doc')}
    assert omissions[('held-copy', 'doc')]['reason'] == 'train_document_shares_held_source_group'


def test_test_copy_wins_same_id_split_conflict_with_explicit_omission():
    train = member('train', {tok: {'doc': identity('train text', split='train')} for tok in TOKENS})
    test = member('test', {tok: {'doc': identity('test text', split='test')} for tok in TOKENS})
    selected, omissions = decide(['train', 'test'], train | test)
    assert selected == {('test', 'doc')}
    omission = omissions[('train', 'doc')]
    assert omission['reason'] == 'conflicting_split_duplicate_id'
    assert omission['kept_member'] == 'test'
    assert omission['kept_text_sha256'] == identity('test text', split='test')['text_sha256']
    assert omission['omitted_text_sha256'] == identity('train text')['text_sha256']


def test_incomplete_test_copy_still_prevents_conflicting_train_copy():
    incomplete_test = member('test', {'lfm': {'doc': identity('held text', split='test', groups=('held',))}})
    train = member('train', {tok: {'doc': identity('train text', split='train', groups=('train',))}
                             for tok in TOKENS})
    selected, omissions = decide(['test', 'train'], incomplete_test | train)
    assert not selected
    assert omissions[('test', 'doc')]['reason'] == 'missing_tokenizer_twin'
    assert omissions[('train', 'doc')]['reason'] == 'conflicting_split_duplicate_id'
    assert omissions[('train', 'doc')]['held_member'] == 'test'


def test_split_conflict_is_retained_as_additional_reason_for_other_omission():
    test = member('test', {tok: {'doc': identity('held text', split='test', groups=('test-group',))}
                           for tok in TOKENS})
    train = member('train', {tok: {'doc': identity('train text', split='train', groups=('blocked',))}
                             for tok in TOKENS})
    selected, omissions = decide(['test', 'train'], test | train, held_groups={'blocked'})
    assert selected == {('test', 'doc')}
    omission = omissions[('train', 'doc')]
    assert omission['reason'] == 'train_document_shares_held_source_group'
    assert any(reason['reason'] == 'conflicting_split_duplicate_id'
               for reason in omission['additional_reasons'])


def test_same_id_distinct_payload_keeps_first_unless_later_declares_supersedes():
    first = member('first', {tok: {'doc': identity('old payload')} for tok in TOKENS})
    later = member('later', {tok: {'doc': identity('new payload')} for tok in TOKENS})
    selected, omissions = decide(['first', 'later'], first | later)
    assert selected == {('first', 'doc')}
    omission = omissions[('later', 'doc')]
    assert omission['reason'] == 'duplicate_document_id'
    assert omission['kept_text_sha256'] == identity('old payload')['text_sha256']
    assert omission['omitted_text_sha256'] == identity('new payload')['text_sha256']

    selected, omissions = decide(['first', 'later'], first | later,
                                 supersedes={'later': {'first'}})
    assert selected == {('later', 'doc')}
    assert omissions[('first', 'doc')]['reason'] == 'superseded_duplicate_document_id'


def test_exact_text_dedup_only_compares_eligible_rows_and_honors_supersedes():
    incomplete = member('incomplete', {'lfm': {'a': identity('shared')}})
    complete = member('complete', {tok: {'b': identity('shared')} for tok in TOKENS})
    selected, omissions = decide(['incomplete', 'complete'], incomplete | complete)
    assert selected == {('complete', 'b')}
    assert omissions[('incomplete', 'a')]['reason'] == 'missing_tokenizer_twin'

    bound = {**identity('shared'), 'source_record_ids': ('same-source',)}
    first = member('first', {tok: {'a': dict(bound)} for tok in TOKENS})
    later = member('later', {tok: {'b': dict(bound)} for tok in TOKENS})
    selected, omissions = decide(['first', 'later'], first | later)
    assert selected == {('first', 'a')}
    assert omissions[('later', 'b')]['reason'] == 'exact_duplicate_document'
    selected, omissions = decide(['first', 'later'], first | later,
                                 supersedes={'later': {'first'}})
    assert selected == {('later', 'b')}
    assert omissions[('first', 'a')]['reason'] == 'superseded_exact_duplicate_document'


def test_equal_text_keeps_distinct_source_or_group_provenance():
    for later_groups, later_source in ((['g2'], 'source-a'), (['g1'], 'source-b')):
        first_row = {'id': 'a', 'split': 'train', 'source_groups': ['g1'],
                     'source_record_ids': ['source-a'], 'text': 'same text'}
        later_row = {'id': 'b', 'split': 'train', 'source_groups': later_groups,
                     'source_record_ids': [later_source], 'text': 'same text'}
        first = member('first', {tok: {'a': _row_identity(first_row)} for tok in TOKENS})
        later = member('later', {tok: {'b': _row_identity(later_row)} for tok in TOKENS})
        selected, omissions = decide(['first', 'later'], first | later)
        assert selected == {('first', 'a'), ('later', 'b')}
        assert not omissions


def test_same_split_but_conflicting_source_groups_are_explicitly_omitted():
    first = member('first', {tok: {'doc': identity('first', groups=('g1',))} for tok in TOKENS})
    later = member('later', {tok: {'doc': identity('later', groups=('g2',))} for tok in TOKENS})
    selected, omissions = decide(['first', 'later'], first | later)
    assert selected == {('first', 'doc')}
    omission = omissions[('later', 'doc')]
    assert omission['reason'] == 'conflicting_split_duplicate_id'
    assert omission['kept_source_groups'] == ['g1']
    assert omission['omitted_source_groups'] == ['g2']


def test_duplicate_ids_inside_one_member_are_ineligible():
    member_rows = {tok: {'doc': identity('text')} for tok in TOKENS}
    selected, omissions = decide(['one'], {'one': member_rows}, duplicates={'one': {'doc'}})
    assert not selected
    assert omissions[('one', 'doc')]['reason'] == 'duplicate_document_id_within_member'


def test_held_duplicate_row_inside_member_still_blocks_conflicting_train_copy():
    rows = {tok: {'doc': identity('train first', groups=('train',))} for tok in TOKENS}
    duplicated_held = identity('held duplicate', split='test', groups=('held',))
    valid_train = member('valid-train', {tok: {'doc': identity('other train', groups=('other',))}
                                         for tok in TOKENS})
    # Pure planner fixtures model duplicate observations explicitly; production fills this argument while scanning.
    selected, omissions = plan_member_rows(
        ['duplicate-member', 'valid-train'], {'duplicate-member': rows} | valid_train,
        {}, TOKENS, set(), set(), member_duplicates={'duplicate-member': {'doc'}},
        member_duplicate_observations={'duplicate-member': {'doc': [duplicated_held]}},
    )
    assert not selected
    assert omissions[('duplicate-member', 'doc')]['reason'] == 'duplicate_document_id_within_member'
    assert omissions[('valid-train', 'doc')]['reason'] == 'conflicting_split_duplicate_id'


def test_target_set_signature_pins_supersedes_policy():
    base = {'id': 'new', 'files': {'full': {'lfm': 'new-lfm', 'mellum': 'new-mellum'}}}
    superseding = {**base, 'supersedes': ['old']}
    assert target_set_manifest.signature([base]) != target_set_manifest.signature([superseding])


def test_target_set_signature_pins_member_cohorts():
    import target_set_manifest as tsm
    member = {'id': 'harness', 'supersedes': [], 'files': {'program-header': {'lfm25': 'a', 'mellum21': 'b'}}}
    plain = tsm.signature([member])
    assert tsm.signature([{**member, 'cohort': None}]) == plain
    assert tsm.signature([{**member, 'cohort': 'agentic-tool-loop'}]) != plain


def test_target_set_signature_changes_when_source_review_holds_change():
    member = {'id': 'render', 'supersedes': [], 'files': {'program-header': {'lfm25': 'a', 'mellum21': 'b'}}}
    assert target_set_manifest.signature([member], 'old-holds') != target_set_manifest.signature([member], 'new-holds')
