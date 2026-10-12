import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

from build_native_turn_union import plan_member_rows  # noqa: E402
import target_set_manifest  # noqa: E402


TOKENS = ['lfm', 'mellum']


def identity(text, split='train', groups=('g1',)):
    return {
        'split': split,
        'source_groups': tuple(sorted(groups)),
        'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }


def member(name, rows_by_tokenizer):
    return {name: {tok: dict(rows) for tok, rows in rows_by_tokenizer.items()}}


def decide(order, members, *, supersedes=None, held_groups=None, held_texts=None, duplicates=None):
    return plan_member_rows(order, members, supersedes or {}, TOKENS,
                            held_groups or set(), held_texts or set(), member_duplicates=duplicates)


def test_incomplete_twin_cannot_claim_same_document_id_or_text():
    older = member('older', {'lfm': {'doc': identity('same')}})
    newer = member('newer', {'lfm': {'doc': identity('same')}, 'mellum': {'doc': identity('same')}})
    selected, omissions = decide(['older', 'newer'], older | newer)
    assert selected == {('newer', 'doc')}
    assert omissions[('older', 'doc')]['reason'] == 'missing_tokenizer_twin'


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

    first = member('first', {tok: {'a': identity('shared')} for tok in TOKENS})
    later = member('later', {tok: {'b': identity('shared')} for tok in TOKENS})
    selected, omissions = decide(['first', 'later'], first | later)
    assert selected == {('first', 'a')}
    assert omissions[('later', 'b')]['reason'] == 'exact_duplicate_document'
    selected, omissions = decide(['first', 'later'], first | later,
                                 supersedes={'later': {'first'}})
    assert selected == {('later', 'b')}
    assert omissions[('first', 'a')]['reason'] == 'superseded_exact_duplicate_document'


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


def test_target_set_signature_pins_supersedes_policy():
    base = {'id': 'new', 'files': {'full': {'lfm': 'new-lfm', 'mellum': 'new-mellum'}}}
    superseding = {**base, 'supersedes': ['old']}
    assert target_set_manifest.signature([base]) != target_set_manifest.signature([superseding])
