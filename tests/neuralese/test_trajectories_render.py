import json

from natlang_neuralese.train.trajectories import crisp_messages, handover_notes, reads, render, target_write

NOTE = "Line 2 is a fee; check prior invoices."
WRITE = {"$write": {"name": "handover:abc", "type": "Neuralese<string>", "source": NOTE}}
RECORD = {"messages": [
    {"role": "system", "content": [{"type": "soft", "name": "prompt:interpreter"}]},
    {"role": "user", "content": [{"type": "soft", "name": "prompt:handover/open"}, {"type": "read", "name": "handover:abc"},
                                 {"type": "soft", "name": "prompt:handover/close"}]},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "compact_history", "arguments": json.dumps({"note": WRITE})}}]},
], "target": {"role": "assistant", "content": "true"}}
TEXTS = {"prompt:interpreter": "You run one call.", "prompt:handover/open": "Your note: ", "prompt:handover/close": "\n\nContinue."}


def test_crisp_rendering_restores_the_original_texts():
    crisp = crisp_messages(RECORD["messages"], TEXTS, handover_notes(RECORD))
    assert crisp[0]["content"] == "You run one call."
    assert crisp[1]["content"] == "Your note: " + NOTE + "\n\nContinue."
    assert json.loads(crisp[2]["tool_calls"][0]["function"]["arguments"]) == {"note": NOTE}
    assert reads(RECORD) == {"handover:abc"} and target_write(RECORD) is None


def test_written_notes_are_block_parts_in_reads_and_inside_the_quoted_argument():
    block = "nz1_" + "b" * 52
    out = render(RECORD["messages"], lambda name: {"type": "neuralese", "id": "nz1_" + "c" * 52}, handover_notes(RECORD),
                 {"handover:abc": block})
    assert {"type": "neuralese", "id": block} in out[1]["content"]
    # Tool arguments stay a JSON string; typed block parts live inside its value.
    arguments = json.loads(out[2]["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"note": [{"type": "neuralese", "id": block, "value_type": "string"}]}


def test_digest_sites_render_as_their_preview_until_the_operator_writes_them():
    messages = [{"role": "tool", "tool_call_id": "scope_0", "content": [
        {"type": "text", "text": "state: unknown = "},
        {"type": "digest", "name": "digest:abc", "source": "{\"a\": 1}", "preview": "{ a: 1, <<cut off: 2 of 3 fields not shown>> }"},
        {"type": "text", "text": "\nDeclared state for the rest of this call."}]}]
    out = render(messages, lambda name: None, {})
    assert out[0]["content"].startswith("state: unknown = { a: 1, <<cut off")


def test_structured_child_returns_keep_their_type_in_crisp_replay():
    value={'source_id':'p0','quote':'An observed fact.'}
    message={'role':'assistant','tool_calls':[{'id':'r','function':{'name':'return_result','arguments':json.dumps({'status':'success','value':{'$write':{'name':'result:p','type':'Neuralese<unknown>','source':json.dumps(value)}}})}}]}
    rendered=render([message],lambda _:None,{})[0]
    assert json.loads(rendered['tool_calls'][0]['function']['arguments'])['value']==value
    message['tool_calls'][0]['function']['arguments']=json.dumps({'status':'success','value':{'$write':{'name':'result:p','type':'Neuralese<string>','source':json.dumps(value)}}})
    rendered=render([message],lambda _:None,{})[0]
    assert json.loads(rendered['tool_calls'][0]['function']['arguments'])['value']==json.dumps(value)


def test_concrete_typed_json_result_replays_as_json_without_parsing_json_strings():
    from natlang_neuralese.train.trajectories import write_value_type
    body = '{"count":4,"flag":true}'
    marker = {"$write": {"name": "typed-result:object", "block_id": "nz1_" + "a" * 32,
                         "type": "Neuralese<{ count: number, flag: boolean }>",
                         "source": body, "source_encoding": "json"}}
    message = {"role": "assistant", "tool_calls": [{"id": "r", "function": {
        "name": "return_result", "arguments": json.dumps({"status": "success", "value": marker})}}]}
    record = {"messages": [], "target": message}
    crisp = render([message], lambda _: None, {})[0]
    assert json.loads(crisp["tool_calls"][0]["function"]["arguments"])["value"] == {"count": 4, "flag": True}
    assert write_value_type(record) == "unknown"

    literal = {"$write": {"name": "typed-result:string", "type": "Neuralese<string>", "source": body}}
    literal_message = {"role": "assistant", "tool_calls": [{"id": "r", "function": {
        "name": "return_result", "arguments": json.dumps({"status": "success", "value": literal})}}]}
    crisp_literal = render([literal_message], lambda _: None, {})[0]
    assert json.loads(crisp_literal["tool_calls"][0]["function"]["arguments"])["value"] == body


def test_nested_structured_child_writes_expand_recursively_in_crisp_replay():
    payload = {"access": "open", "items": ["OH-9", {"speaker": "Aroha Lane"}]}
    nested = {"status": "success", "value": {
        "record": {"$write": {"name": "result:nested", "type": "Neuralese<unknown>",
                               "source": json.dumps(payload)}},
        "labels": [{"$write": {"name": "result:string", "type": "Neuralese<string>",
                                "source": "catalog verified"}}],
    }}
    message = {"role": "assistant", "tool_calls": [{"id": "r", "function": {
        "name": "return_result", "arguments": json.dumps(nested)}}]}

    crisp = render([message], lambda _: None, {})[0]
    args = json.loads(crisp["tool_calls"][0]["function"]["arguments"])
    assert args == {"status": "success", "value": {
        "record": payload,
        "labels": ["catalog verified"],
    }}

    blocks = {"result:nested": "nz1_" + "a" * 52, "result:string": "nz1_" + "b" * 52}
    replay = render([message], lambda _: None, {}, blocks=blocks)[0]
    args = json.loads(replay["tool_calls"][0]["function"]["arguments"])
    assert args == {"status": "success", "value": {
        "record": [{"type": "neuralese", "id": blocks["result:nested"], "value_type": "unknown"}],
        "labels": [[{"type": "neuralese", "id": blocks["result:string"], "value_type": "string"}]],
    }}


def test_nested_writer_site_keeps_exact_path_source_type_and_template_cut():
    from natlang_neuralese.train.trajectories import (
        handover_notes, native_writer_prefix, target_writes, write_site,
        write_site_arguments, write_value_path, write_value_type,
    )
    source = "open with buffer"
    raw_args = {"status": "success", "value": {
        "site": "JUNIPER-4", "access": {"$write": {
            "name": "result:access", "type": "Neuralese<string>", "source": source}},
    }}
    record = {"messages": [], "target": {"tool_calls": [{"function": {
        "name": "return_result", "arguments": json.dumps(raw_args)}}]}}
    assert target_writes(record) == {"result:access"}
    assert handover_notes(record) == {"result:access": source}
    assert write_site(record) == ("return_result", {"status": "success"}, "value", "result:access")
    assert write_value_path(record, "result:access") == ("value", "access")
    assert write_site_arguments(record, "result:access") == raw_args
    assert write_value_type(record) == "string"

    def template(messages, generation):
        prefix = "USER x\nASSISTANT "
        if generation:return prefix
        call = messages[-1]["tool_calls"][0]["function"]
        return prefix + call["name"] + "(" + json.dumps(call["arguments"], separators=(",", ":")) + ") END"

    producer = {**record, "_active_write_name": "result:access"}
    cut = native_writer_prefix(producer, template)
    assert cut.endswith('"access":"')
    assert '"site":"JUNIPER-4"' in cut


def test_nested_unknown_writer_site_keeps_json_value_without_string_coercion():
    from natlang_neuralese.train.trajectories import handover_notes, write_value_type
    payload = {"source_id": "p0", "quote": "An observed fact."}
    marker = {"$write": {"name": "result:object", "type": "Neuralese<unknown>",
                         "source": json.dumps(payload)}}
    record = {"messages": [], "target": {"tool_calls": [{"function": {"name": "return_result",
        "arguments": json.dumps({"status": "success", "value": {"evidence": [marker]}})}}]}}
    assert handover_notes(record) == {"result:object": json.dumps(payload)}
    assert write_value_type(record) == "unknown"


def test_nested_write_site_paths_are_unique_and_ambiguous_names_are_rejected():
    import pytest
    from natlang_neuralese.train.trajectories import write_site, write_value_path
    def marker(name, source):
        return {"$write": {"name": name, "type": "Neuralese<string>", "source": source}}
    arguments = {"status": "success", "value": {"access": marker("result:access", "open"),
                                                     "decision": marker("result:decision", "allow")}}
    record = {"messages": [], "target": {"tool_calls": [{"function": {
        "name": "return_result", "arguments": json.dumps(arguments)}}]}}
    assert write_value_path(record, "result:access") == ("value", "access")
    assert write_value_path(record, "result:decision") == ("value", "decision")
    active = {**record, "_active_write_name": "result:decision"}
    assert write_site(active)[3] == "result:decision"

    ambiguous = json.loads(json.dumps(record))
    ambiguous["target"]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "value": {"access": marker("duplicate", "open"), "decision": marker("duplicate", "closed")}})
    with pytest.raises(ValueError, match="duplicate writer name"):
        write_site(ambiguous)
    with pytest.raises(ValueError, match="multiple value paths"):
        write_value_path(ambiguous, "duplicate")
