import pytest
from scripts.neuralese_data.review_judgment_shortcuts import (JudgmentCodeChecker, judgment_signature, review,
                                                              shortcut)


def test_judgment_signatures_are_typed_natural_language_decisions():
    assert judgment_signature('nl@eval:5(note: string): boolean')
    assert judgment_signature('nl(input: string): "billing" | "fees" | "other"')
    assert judgment_signature('nl@eval:2(id: string, text: string): { id: string, decision: "a" | "b" }')
    assert not judgment_signature('nl@eval:8(item: Item): string')
    assert not judgment_signature('summarize(doc: string): boolean')
    assert not judgment_signature('nl@eval:3(row: Row): { amount: number }')


def test_policy_holds_prose_tests_and_clears_exact_parsing():
    assert shortcut({'use': 'test', 'pattern': 'fee|charge|cost'})
    assert shortcut({'use': 'test', 'pattern': 'bug'})
    assert not shortcut({'use': 'extraction', 'pattern': 'refund'})
    assert not shortcut({'use': 'test', 'pattern': 'Status:\\s*(\\w+)'})
    assert not shortcut({'use': 'test', 'pattern': '^\\d{4}-\\d{2}-\\d{2}$'})
    assert not shortcut({'use': 'test', 'pattern': '.md'})
    assert not shortcut({'use': 'test', 'pattern': '^(true|yes)$'})
    assert not shortcut({'use': 'test', 'pattern': 'alarm', 'code': 'options.includes("alarm")'})
    assert shortcut({'use': 'test', 'pattern': 'alarm', 'code': 'text.includes("alarm")'})


def test_actual_typescript_evidence_separates_tests_from_extraction():
    code = ('const fee = /fee|charge/i.test(input.question);\n'
            'const amount = input.text.match(/Total: (\\d+)/);\n'
            'const bug = input.feedback.includes("bug");\n'
            'const path = "a/b/c"; // not /a regex/\n'
            'return fee || bug;')
    with JudgmentCodeChecker() as checker:
        found = checker(code)
        held = list(review([{'id': 'x', 'signature': 'nl@eval:1(input: In): boolean', 'codes': [code]},
                            {'id': 'y', 'signature': 'nl@eval:1(input: In): { amount: number }', 'codes': [code]},
                            {'id': 'z', 'signature': 'nl@eval:1(input: In): boolean',
                             'codes': ['return input.text.split(/\\n/).length > 2;']}], checker))
    kinds = {(item['kind'], item['use'], item['pattern']) for item in found['evidence']}
    assert ('regex', 'test', 'fee|charge') in kinds and ('regex', 'extraction', 'Total: (\\d+)') in kinds
    assert ('keyword', 'test', 'bug') in kinds and found['parse_errors'] == 0
    assert [item['disposition'] for item in held] == ['hold', 'clear', 'clear']
    assert {item['pattern'] for item in held[0]['evidence']} == {'fee|charge', 'bug'}
