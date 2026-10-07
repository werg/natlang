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
