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


def soft_capture_example():
    block_id = "nz1_" + "a" * 52
    body = "Apply the supplied rule to the note."
    code_source = f"\ue000{block_id}\ue001"
    prefix = "const judge: Neuralese<(note: string) => Promise<boolean>> = nl.with({ policy })<(note: string) => Promise<boolean>>`"
    suffix = "`; return await judge(note);"
    code = prefix + code_source + suffix
    binding = {
        "schema": "natlang.inline-capture-binding-plan/1",
        "syntax": "nl.with",
        "body_block_id": block_id,
        "body_source_sha256": sha(body),
        "parent_invocation_id": "parent-1",
        "parent_scope_sha256": "1" * 64,
        "child_scope_sha256": "2" * 64,
        "captures": [{"name": "policy", "type": "string", "mode": "snapshot", "value": "Only use the note."}],
    }
    metadata = sidecar(code, [
        {"type": "text", "text": prefix},
        {"$write": {"name": "site", "type": INLINE_WRITE_TYPE, "source": body, "code_source": code_source}},
        {"type": "text", "text": suffix},
    ])
    metadata["sites"] = [{"name": "site", "plan": {"capture_binding_plan": binding}}]
    raw = json.dumps({"code": code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    return raw, metadata, body, code_source


def test_explicit_snapshot_body_plan_validates_and_crisp_code_bytes_are_preserved():
    raw, metadata, body, code_source = soft_capture_example()
    checked = validate_inline_instruction_code(raw, metadata)
    assert checked.valid, checked.reason
    assert checked.value.writes[0].source == body
    assert checked.value.writes[0].code_source == code_source
    assert render_inline_instruction_code(checked, {}) == json.loads(raw)["code"]
    assert render_inline_instruction_arguments(raw, metadata, {}) == raw
    prefix, source = inline_instruction_prefix_body(checked, "site")
    assert prefix.endswith("nl.with({ policy })<(note: string) => Promise<boolean>>`" )
    assert source == body
    rendered = render_inline_instruction_code(checked, {"site": "nz1_" + "b" * 52})
    assert rendered[0]["text"] == prefix
    assert rendered[1] == {"type": "neuralese", "id": "nz1_" + "b" * 52, "value_type": "string"}
    assert rendered[2]["text"].endswith("`; return await judge(note);")


def test_explicit_snapshot_body_plan_rejects_bad_bindings_or_prefix_syntax():
    raw, metadata, _, _ = soft_capture_example()
    bad_plan = json.loads(json.dumps(metadata))
    bad_plan["sites"][0]["plan"]["capture_binding_plan"]["captures"][0]["mode"] = "live"
    assert validate_inline_instruction_code(raw, bad_plan).reason == "capture-binding-plan-invalid"
    bad_prefix = json.loads(json.dumps(metadata))
    changed_code = json.loads(raw)["code"].replace("nl.with({ policy })", "nl.with({ other })")
    changed_raw = json.dumps({"code": changed_code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    bad_prefix["code_sha256"] = sha(changed_code)
    bad_prefix["parts"][0]["text"] = bad_prefix["parts"][0]["text"].replace("nl.with({ policy })", "nl.with({ other })")
    assert validate_inline_instruction_code(changed_raw, bad_prefix).reason == "explicit-with-prefix-mismatch"
    malformed = json.loads(json.dumps(metadata))
    malformed["sites"][0]["plan"]["capture_binding_plan"]["captures"][0]["type"] = "object"
    assert validate_inline_instruction_code(raw, malformed).reason == "capture-binding-plan-invalid"


def schema2_capture_example(*, body_kind="literal", prefix_style="after", capture_values=None):
    body = "Apply the supplied policy to the note."
    block_id = "nz1_" + "c" * 52
    code_source = body if body_kind == "literal" else f"<|neuralese|>{body}<|/neuralese|>"
    if capture_values is None:
        capture_values = [("policy", "string", "Only use the policy.", "local"),
                          ("attempts", "number", 3, "input"), ("enabled", "boolean", True, "block")]
    names = [item[0] for item in capture_values]
    captures_text = ", ".join(names)
    function_type = "<(note: string) => Promise<boolean>>"
    if prefix_style == "before":
        prefix = f"const judge = nl.with{function_type}({{ {captures_text} }})`"
    else:
        prefix = f"const judge = nl.with({{ {captures_text} }}){function_type}`"
    suffix = "`; return await judge(note);"
    code = prefix + code_source + suffix
    template_span = {"file": "eval", "start": len(prefix) - 1,
                     "end": len(prefix) + len(code_source) + 1}
    runtime_code = (code.replace(code_source, f"\ue000{block_id}\ue001", 1)
                    if body_kind == "neuralese_block" else code)
    creation = {"parentInvocationId": "parent-1", "toolCallId": "call-parent-1", "actionOrdinal": 2,
                "writtenCodeSha256": sha(runtime_code), "checkedCodeSha256": sha("checked:" + code),
                "definitionId": "nl:test-site", "sourceSpan": {"file": "eval", "start": 0, "end": len(code)},
                "templateSpan": template_span, "checkedTemplateSpan": template_span}

    def canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    binding_captures = []
    for name, value_type, value, source in capture_values:
        primitive = {"type": value_type, "value": value}
        value_canonical = canonical(primitive)
        snapshot = {"name": name, "type": value_type, "source": source, "value": value, "mode": "snapshot",
                    "value_canonical": value_canonical,
                    "value_sha256": hashlib.sha256(b"natlang.inline-capture-snapshot/v1\0" + value_canonical.encode()).hexdigest(),
                    "creation": creation}
        binding_captures.append({"name": name, "type": value_type, "mode": "snapshot", "value": value,
                                 "source": source, "host_snapshot": snapshot})
    binding = {"schema": "natlang.inline-capture-binding-plan/2", "syntax": "nl.with",
               "body_kind": body_kind, **({"body_block_id": block_id} if body_kind == "neuralese_block" else {}),
               "creation": creation, "body_source_sha256": sha(body), "captures": binding_captures,
               "parent_invocation_id": "parent-1", "parent_scope_sha256": "1" * 64,
               "child_scope_sha256": "2" * 64}
    metadata = sidecar(code, [
        {"type": "text", "text": prefix},
        {"$write": {"name": "site", "type": INLINE_WRITE_TYPE, "source": body, "code_source": code_source}},
        {"type": "text", "text": suffix},
    ])
    metadata["sites"] = [{"name": "site", "plan": {"definition_id": "nl:test-site", "capture_binding_plan": binding}}]
    raw = json.dumps({"code": code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    return raw, metadata, body, code_source


def test_schema2_primitive_capture_attestation_supports_literal_and_block_bodies_and_both_prefixes():
    for body_kind in ("literal", "neuralese_block"):
        for prefix_style in ("before", "after"):
            raw, metadata, body, code_source = schema2_capture_example(body_kind=body_kind, prefix_style=prefix_style)
            checked = validate_inline_instruction_code(raw, metadata)
            assert checked.valid, checked.reason
            prefix, recovered_body = inline_instruction_prefix_body(checked, "site")
            assert recovered_body == body
            assert prefix.endswith("`")
            assert render_inline_instruction_code(checked, {}) == json.loads(raw)["code"]
            rendered = render_inline_instruction_code(checked, {"site": "nz1_" + "d" * 52})
            assert rendered[1] == {"type": "neuralese", "id": "nz1_" + "d" * 52, "value_type": "string"}
            assert code_source in json.loads(raw)["code"]


def test_schema2_snapshot_validation_is_bound_to_creation_code_and_primitive_payload():
    raw, metadata, _, _ = schema2_capture_example()
    assert validate_inline_instruction_code(raw, metadata).valid

    def altered(mutator):
        result = json.loads(json.dumps(metadata))
        mutator(result["sites"][0]["plan"]["capture_binding_plan"])
        assert validate_inline_instruction_code(raw, result).reason == "capture-binding-plan-invalid"

    altered(lambda binding: binding["creation"].update(parentInvocationId="other-parent"))
    altered(lambda binding: binding["creation"].update(definitionId="nl:other"))
    altered(lambda binding: binding["captures"][0]["host_snapshot"].update(creation={"forged": True}))
    altered(lambda binding: binding["captures"][0]["host_snapshot"].update(value="changed"))
    altered(lambda binding: binding["captures"][0]["host_snapshot"].update(value_sha256="0" * 64))
    altered(lambda binding: binding["captures"][0].pop("host_snapshot"))
    altered(lambda binding: binding.update(body_block_id="nz1_" + "d" * 52))
    def set_bad_number(binding, index, value):
        capture = binding["captures"][index]
        snapshot = capture["host_snapshot"]
        capture["value"] = snapshot["value"] = value
        # Keep the attestation internally consistent; the value itself must be rejected.
        canonical = json.dumps({"type": "number", "value": 0 if value == 0 else value},
                               ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        snapshot["value_canonical"] = canonical
        snapshot["value_sha256"] = hashlib.sha256(
            b"natlang.inline-capture-snapshot/v1\0" + canonical.encode()).hexdigest()

    altered(lambda binding: set_bad_number(binding, 1, -0.0))
    altered(lambda binding: set_bad_number(binding, 1, float("inf")))

    changed_code = json.loads(raw)["code"].replace("return await", "return await /*changed*/")
    changed_raw = json.dumps({"code": changed_code, "finish": True}, ensure_ascii=False, separators=(",", ":"))
    changed_sidecar = json.loads(json.dumps(metadata))
    changed_sidecar["code_sha256"] = sha(changed_code)
    changed_sidecar["parts"][0]["text"] = changed_sidecar["parts"][0]["text"].replace("return await", "return await /*changed*/")
    assert validate_inline_instruction_code(changed_raw, changed_sidecar).reason == "capture-binding-plan-invalid"


def test_schema2_snapshot_uses_host_canonical_number_payload_without_python_reformatting():
    for value, spelling in ((1e-7, "1e-7"), (1e-6, "0.000001"), (1e21, "1e+21"), (0.10000000000000002, "0.10000000000000002")):
        raw, metadata, _, _ = schema2_capture_example()
        binding = metadata["sites"][0]["plan"]["capture_binding_plan"]
        capture = binding["captures"][1]
        snapshot = capture["host_snapshot"]
        capture["value"] = snapshot["value"] = value
        canonical = '{"type":"number","value":' + spelling + '}'
        snapshot["value_canonical"] = canonical
        snapshot["value_sha256"] = hashlib.sha256(
            b"natlang.inline-capture-snapshot/v1\0" + canonical.encode()).hexdigest()
        assert validate_inline_instruction_code(raw, metadata).valid, spelling

    raw, metadata, _, _ = schema2_capture_example()
    binding = metadata["sites"][0]["plan"]["capture_binding_plan"]
    snapshot = binding["captures"][1]["host_snapshot"]
    for malformed in ('{"type":"number","value":1,"value":1}',
                      '{"type":"number","value":NaN}',
                      '{"type":"number","value":-0}'):
        snapshot["value_canonical"] = malformed
        snapshot["value_sha256"] = hashlib.sha256(
            b"natlang.inline-capture-snapshot/v1\0" + malformed.encode()).hexdigest()
        assert validate_inline_instruction_code(raw, metadata).reason == "capture-binding-plan-invalid"



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
