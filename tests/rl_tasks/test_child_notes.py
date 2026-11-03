"""Reference notes for Neuralese producer children (scripts/rl_tasks/child_notes.py, sources/composed_notes.py):
the entries reach exactly the child call they are about through rl/privileged_context.applicable, and held cases stay
untouched. No upstream data: items are small synthetic dicts of the loaders' normalized shape."""
import json

from natlang_neuralese.rl.privileged_context import Fact, _child_match, _child_text, applicable
from scripts.rl_tasks import child_notes as cn
from scripts.rl_tasks.sources import composed, composed_notes as notes

QUESTION = 'Who developed the eponymous character from the series that contains "Mickey\'s Safari"?'


def opening(signature, **arguments):
    """A child prompt as the runtime renders it: the signature line, then the declared argument values."""
    lines = [f'You are inside this call: {signature}', '', 'Instructions:', 'Write a note.', '']
    lines += [f'{name}: {type_} = {value}' for name, (type_, value) in arguments.items()]
    return '\n'.join(lines)


def reaches(entry, text):
    fact = Fact('child_value:0', 'child_value', _child_text(entry), match=_child_match(entry))
    return applicable(fact, {'name': 'c', 'role': 'child', 'text': text})


def musique_item():
    documents = [{'title': 'Mickey Mouse', 'text': 'Mickey Mouse is a cartoon character. He was created by Walt Disney in 1928.'},
                 {'title': 'Letterland', 'text': "Mickey's Safari in Letterland is a video game in the Mickey Mouse series."},
                 {'title': 'Unrelated', 'text': 'Nothing here.'}]
    return {'question': QUESTION, 'answer': 'Walt Disney', 'documents': documents,
            'steps': [{'question': "What series is Mickey's Safari in Letterland from?", 'answer': 'Mickey Mouse', 'support': 1},
                      {'question': 'Who developed #1 ?', 'answer': 'Walt Disney', 'support': 0}],
            'support_paragraph': {0: 'Mickey Mouse', 1: 'Letterland', 2: 'Unrelated'}}


def test_fragments_are_written_as_the_runtime_renders_values():
    assert cn.fragment('He said "hi"\nthen left, quickly and quietly for the day', limit=24) == 'He said \\"hi\\"\\nthen left,'
    assert cn.fragment('abc') is None
    assert cn.document_key('A "B"') == '{ title: "A \\"B\\"", text: "'
    assert cn.string_key('person', 'Dale') == 'person: string = "Dale"'


def test_a_gold_note_reaches_its_own_child_call_only():
    item = musique_item()
    program = composed.RELAY_GATHER
    entries = notes.musique_relay(program, item)
    assert [e['match'][0] for e in entries] == ['You are inside this call: gather(', 'You are inside this call: refine(']
    assert all(e['provenance'] in ('gold', 'checked') for e in entries)
    gather = entries[0]
    assert 'Mickey Mouse' in gather['value'] and 'Walt Disney' in gather['value'] and 'Letterland' in gather['value']
    assert 'The chain reaches the answer: Walt Disney.' in entries[1]['value']
    call = opening('gather(question: string, documents: { title: string, text: string }[]): Neuralese<string>',
                   question=('string', json.dumps(QUESTION, ensure_ascii=False)), documents=('X', '[]'))
    assert reaches(gather, call)
    assert not reaches(gather, call.replace('gather(', 'refine('))
    other = call.replace(json.dumps(QUESTION)[1:30], 'Another question about somet')
    assert not reaches(gather, other)


def test_per_document_children_are_pinned_by_the_document_title():
    item = {**musique_item()}
    entries = notes.musique_passage_notes(item)
    assert {e['match'][1] for e in entries} == {cn.document_key('Mickey Mouse'), cn.document_key('Letterland')}
    mickey = next(e for e in entries if 'Mickey Mouse"' in e['match'][1])
    text = opening('read_passage(step: string, document: { title: string, text: string }): Neuralese<string>',
                   step=('string', '"Who developed it?"'),
                   document=('{...}', '{ title: "Mickey Mouse", text: "Mickey Mouse is a cartoon ... Letterland ..." }'))
    assert reaches(mickey, text)
    assert not reaches(mickey, text.replace('title: "Mickey Mouse"', 'title: "Letterland"'))
    assert 'He was created by Walt Disney in 1928.' in mickey['value'] and mickey['provenance'] == 'checked'


def test_typed_profiles_fit_their_declared_neuralese_type():
    item = {'question': 'Which film came out first, Alpha or Beta?', 'documents': [
        {'title': 'Alpha (film)', 'text': 'Alpha is a 1999 film.'}, {'title': 'Gamma', 'text': 'Other.'}],
        'evidences': [['Alpha', 'publication date', '1999']], 'support': [('Alpha (film)', 0)]}
    entries = notes.wiki_profiles(item)
    assert [e['value'] for e in entries] == [
        {'subject': 'Alpha (film)', 'relevant': True, 'facts': [{'attribute': 'publication date', 'value': '1999'}]},
        {'subject': 'Gamma', 'relevant': False, 'facts': []}]
    signature = ('profile(question: string, document: { title: string, text: string }): Neuralese<{ subject: string, '
                 'relevant: boolean, facts: { attribute: string, value: string }[] }>')
    call = opening(signature, question=('string', json.dumps(item['question'])),
                   document=('{...}', '{ title: "Alpha (film)", text: "Alpha is a 1999 film." }'))
    assert reaches(entries[0], call) and not reaches(entries[1], call)
    # A value of another shape than the declared result is never offered to the child.
    broken = dict(entries[0], value='a string where a record is declared')
    assert not reaches(broken, call)


def test_a_babi_state_note_holds_the_supporting_facts_read_so_far():
    item = {'facts': ['1. Mary went to the kitchen.', '2. John went to the garden.', '3. Mary took the milk.',
                      '4. Mary went to the hall.'], 'support': [1, 4]}
    chunks = [item['facts'][:2], item['facts'][2:]]
    first, second = notes.babi_states(item, chunks)
    assert first['match'][0].endswith('begin_state(') and second['match'][0].endswith('advance(')
    assert first['value'] == 'Facts so far that bear on the question:\n1. Mary went to the kitchen.'
    assert second['value'].endswith('4. Mary went to the hall.') and first['provenance'] == 'gold'


def test_trivia_candidates_carry_the_checked_answer_sentence_from_the_first_section_that_holds_it():
    item = {'answer': 'John and Charles Wesley', 'aliases': []}
    sections = [{'title': 'A', 'text': 'Methodism began in England.'},
                {'title': 'A', 'text': 'It was started by John and Charles Wesley. They were brothers.'}]
    first, second = notes.trivia_candidates(item, sections)
    assert first['value'] == {'candidates': []}
    assert second['value']['candidates'][0]['answer'] == 'John and Charles Wesley'
    assert 'brothers' not in second['value']['candidates'][0]['evidence'] and second['provenance'] == 'checked'


def test_clutrr_relations_come_only_from_proof_facts_the_sentence_names():
    item = {'story': "[Ashley]'s daughter, [Lillian], asked for a story. Dinner was ready.",
            'proof': "[{('Ashley', 'son', 'Nicholas'): [('Ashley', 'daughter', 'Lillian'), ('Lillian', 'brother', 'Nicholas')]}]"}
    entries = notes.clutrr_relations(item)
    assert [e['value'] for e in entries] == [[{'person': 'Ashley', 'relation': 'daughter', 'relative': 'Lillian'}]]
    assert entries[0]['provenance'] == 'checked'


def test_only_training_cases_get_children_and_coverage_counts_declared_producers():
    case = {'split': 'train', 'curriculum': {'reference': {'root': []}}}
    held = {'split': 'dev', 'curriculum': {'reference': {'root': []}}}
    entry = cn.child('gather', ['question text'], 'note', 'gold')
    assert cn.attach(held, [entry])['curriculum']['reference'] == {'root': []}
    assert cn.attach(case, [entry, None])['curriculum']['reference']['children'] == [entry]
    case['semantics'] = {'root': 'r.nl', 'files': {
        'r.nl': '---\nargs: {}\nreturns: "string"\n---\nbody', 'r/gather.nl': '---\nargs: {}\nreturns: "Neuralese<string>"\n---\nb',
        'r/refine.nl': '---\nargs: {}\nreturns: "Neuralese<string>"\n---\nb'}}
    assert cn.coverage([case]) == {'sites': 2, 'covered': 1, 'uncovered': 1, 'provenance': {'gold': 1}}
    try:
        cn.child('gather', ['x'], 'note', 'invented')
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown provenance must be rejected')
