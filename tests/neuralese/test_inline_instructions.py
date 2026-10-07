import hashlib
import json

from natlang_neuralese.train.inline_instructions import (
    INLINE_CODE_SCHEMA,
    INLINE_WRITE_TYPE,
    inline_instruction_prefix_body,
    render_inline_instruction_arguments,
    render_inline_instruction_code,
    validate_inline_instruction_code,
)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sidecar(code, parts):
    return {"schema": INLINE_CODE_SCHEMA, "code_sha256": sha(code), "parts": parts}


def two_site_example():
    first = "First prompt."
    second = "Second prompt."
    code = ("const first = nl<boolean>`First prompt.`; const second = nl<boolean>"
            "`Second prompt.`; return [first, second];")
    parts = [
        {"type": "text", "text": "const first = nl<boolean>`"},
        {"$write": {"name": "inline-site:a", "type": INLINE_WRITE_TYPE, "source": first}},
        {"type": "text", "text": "`; const second = nl<boolean>`"},
        {"$write": {"name": "inline-site:b", "type": INLINE_WRITE_TYPE, "source": second}},
        {"type": "text", "text": "`; return [first, second];"},
    ]
    arguments = {"code": code, "finish": True}
    raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return arguments, raw_arguments, sidecar(code, parts)


def test_validates_multiple_writes_and_prefix_for_later_writer_keeps_earlier_body_crisp():
    arguments, raw, metadata = two_site_example()
    checked = validate_inline_instruction_code(raw, metadata)
    assert checked.valid
    assert checked.value.code == arguments["code"]
    assert [write.name for write in checked.value.writes] == ["inline-site:a", "inline-site:b"]
    assert inline_instruction_prefix_body(checked, "inline-site:a") == (
        "const first = nl<boolean>`", "First prompt."
    )
    assert inline_instruction_prefix_body(checked, "inline-site:b") == (
        "const first = nl<boolean>`First prompt.`; const second = nl<boolean>`", "Second prompt."
    )


def test_crisp_render_preserves_original_code_and_argument_json_bytes():
    _, raw, metadata = two_site_example()
    checked = validate_inline_instruction_code(raw, metadata)
    assert render_inline_instruction_code(checked, {}) == json.loads(raw)["code"]
    assert render_inline_instruction_arguments(raw, metadata, {}) == raw


def test_selected_blocks_render_existing_engine_parts_for_one_or_all_sites():
    _, raw, metadata = two_site_example()
    one = render_inline_instruction_code(validate_inline_instruction_code(raw, metadata),
                                         {"inline-site:b": "nz1_second"})
    assert one == [
        {"type": "text", "text": "const first = nl<boolean>`First prompt.`; const second = nl<boolean>`"},
        {"type": "neuralese", "id": "nz1_second", "value_type": "string"},
        {"type": "text", "text": "`; return [first, second];"},
    ]
    all_sites = render_inline_instruction_code(validate_inline_instruction_code(raw, metadata),
                                               {"inline-site:a": "nz1_first", "inline-site:b": "nz1_second"})
    assert [part["id"] for part in all_sites if part["type"] == "neuralese"] == ["nz1_first", "nz1_second"]
    assert "First prompt." not in "".join(part.get("text", "") for part in all_sites)


def test_rendered_arguments_keep_non_code_fields_and_put_parts_only_in_code():
    _, raw, metadata = two_site_example()
    rendered = render_inline_instruction_arguments(raw, metadata, {"inline-site:a": "nz1_first"})
    parsed = json.loads(rendered)
    assert parsed["finish"] is True
    assert isinstance(parsed["code"], list)
    assert {part.get("id") for part in parsed["code"] if part["type"] == "neuralese"} == {"nz1_first"}


def test_crisp_bytes_outside_writer_body_are_reconstructed_exactly():
    prefix = 'const escaped = "\\u263A\\n";\nconst judge = nl<boolean>`'
    body = "Does the note describe a current outage?"
    suffix = '`; return await judge(note);'
    code = prefix + body + suffix
    metadata = sidecar(code, [
        {"type": "text", "text": prefix},
        {"$write": {"name": "site", "type": INLINE_WRITE_TYPE, "source": body}},
        {"type": "text", "text": suffix},
    ])
    raw = json.dumps({"code": code}, ensure_ascii=False, separators=(",", ":"))
    checked = validate_inline_instruction_code(raw, metadata)
    assert checked.valid
    assert render_inline_instruction_code(checked, {}) == code
    assert render_inline_instruction_arguments(raw, metadata, {}) == raw


def test_forged_write_source_hash_and_missing_arguments_fail_closed_without_partial_decode():
    _, raw, metadata = two_site_example()
    forged = json.loads(json.dumps(metadata))
    forged["parts"][1]["$write"]["source"] = "forged prompt"
    assert validate_inline_instruction_code(raw, forged).reason == "source-reconstruction-mismatch"
    assert render_inline_instruction_code(validate_inline_instruction_code(raw, forged),
                                          {"inline-site:a": "nz1_a"}) is None
    wrong_hash = dict(metadata, code_sha256="0" * 64)
    assert validate_inline_instruction_code(raw, wrong_hash).reason == "code-hash-mismatch"
    assert validate_inline_instruction_code(json.dumps({"finish": True}), metadata).reason == "code-argument-missing"
    assert render_inline_instruction_arguments(raw, wrong_hash, {"inline-site:a": "nz1_a"}) is None


def test_wrong_schema_duplicate_writer_or_invalid_member_invalidates_entire_sidecar():
    _, raw, metadata = two_site_example()
    wrong_schema = dict(metadata, schema="future-version")
    assert not validate_inline_instruction_code(raw, wrong_schema).valid
    duplicate = json.loads(json.dumps(metadata))
    duplicate["parts"][3]["$write"]["name"] = "inline-site:a"
    assert validate_inline_instruction_code(raw, duplicate).reason == "writer-name-missing-or-duplicate"
    malformed = json.loads(json.dumps(metadata))
    malformed["parts"][3]["$write"]["type"] = "Neuralese<unknown>"
    assert validate_inline_instruction_code(raw, malformed).reason == "writer-contract-invalid"


def test_escaped_or_interpolated_bodies_are_held():
    for body in (r"escaped\n", r"escaped\`", "${dynamic}"):
        code = "const judge = nl<boolean>`" + body + "`;"
        metadata = sidecar(code, [
            {"type": "text", "text": "const judge = nl<boolean>`"},
            {"$write": {"name": "site", "type": INLINE_WRITE_TYPE, "source": body}},
            {"type": "text", "text": "`;"},
        ])
        raw = json.dumps({"code": code})
        checked = validate_inline_instruction_code(raw, metadata)
        assert not checked.valid
        assert checked.value is None



def test_actual_trajectory_rendering_enumerates_all_writers_and_preserves_gold_producer_bodies():
    from natlang_neuralese.train.trajectories import (render,handover_notes,write_site,
        target_writes,inline_write_prefix,producer_text_target)
    _,raw,metadata=two_site_example()
    call={'id':'teacher_0_0','function':{'name':'eval','arguments':raw},'neuralese_code':metadata}
    target={'role':'assistant','content':'','tool_calls':[call]}
    record={'messages':[],'target':target}
    assert target_writes(record)=={'inline-site:a','inline-site:b'}
    selected={**record,'_active_write_name':'inline-site:b'}
    assert write_site(selected)==('eval',{},'code','inline-site:b')
    assert inline_write_prefix(selected,'inline-site:b').endswith('const second = nl<boolean>`')
    assert handover_notes(record)=={'inline-site:a':'First prompt.','inline-site:b':'Second prompt.'}
    blocks={'inline-site:a':'nz1_'+'a'*52,'inline-site:b':'nz1_'+'b'*52}
    crisp=render([target],lambda name:None,handover_notes(record))[0]
    assert crisp['tool_calls'][0]['function']['arguments']==raw
    assert 'neuralese_code' not in crisp['tool_calls'][0]
    written=render([target],lambda name:None,handover_notes(record),blocks)[0]
    code=json.loads(written['tool_calls'][0]['function']['arguments'])['code']
    assert [part['id'] for part in code if part['type']=='neuralese']==list(blocks.values())
    gold=producer_text_target(record,{},blocks)
    assert gold['tool_calls'][0]['function']['arguments']==raw


def test_trajectory_sidecar_validation_never_renders_a_corrupted_partial_code():
    import pytest
    from natlang_neuralese.train.trajectories import render,write_sites
    _,raw,metadata=two_site_example()
    metadata['code_sha256']='bad'
    target={'role':'assistant','tool_calls':[{'function':{'name':'eval','arguments':raw},'neuralese_code':metadata}]}
    with pytest.raises(ValueError,match='invalid inline instruction'):
        render([target],lambda name:None,{})
    with pytest.raises(ValueError,match='invalid inline instruction'):
        write_sites({'target':target})


def test_native_writer_prefix_stops_at_each_actual_body_in_native_argument_syntax():
    from natlang_neuralese.train.trajectories import native_writer_prefix,inline_write_prefix
    _,raw,metadata=two_site_example()
    record={'target':{'role':'assistant','tool_calls':[{'function':{'name':'eval','arguments':raw},'neuralese_code':metadata}]}}
    def template(messages,generate):
        prefix='USER x ASSISTANT '
        if generate:return prefix
        call=messages[-1]['tool_calls'][0]['function']
        arguments=', '.join(f'{key}={json.dumps(value,ensure_ascii=False)}' for key,value in call['arguments'].items())
        return prefix+call['name']+'('+arguments+') END'
    for name in ['inline-site:a','inline-site:b']:
        producer={**record,'_active_write_name':name}
        prefix=native_writer_prefix(producer,template)
        expected='eval(code="'+json.dumps(inline_write_prefix(producer,name),ensure_ascii=False)[1:-1]
        assert prefix==expected
        assert prefix.endswith('`')
