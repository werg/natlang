from scripts.neuralese_data import prompt_variant_render as variants


def test_header_preserves_supplied_tools_and_packet_options(monkeypatch):
    context = [{"role": "system", "content": "program-specific instruction"},
               {"role": "user", "content": "work"}]
    tools = [{"type": "function", "function": {"name": "return_result",
              "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}}]
    target = {"role": "assistant", "tool_calls": [{"id": "call-1"}]}
    seen = {}
    def packet(tokenizer, turns, reply, schemas, **options):
        seen.update(turns=turns, reply=reply, schemas=schemas, options=options)
        return "packet"
    monkeypatch.setattr(variants, "native_gold_packet", packet)
    assert variants.render_variant(object(), context, target, tools, "program-header",
                                   return_offsets=True) == "packet"
    assert seen["schemas"] == tools
    assert seen["reply"] == target
    assert seen["options"] == {"prompt_rendering": "full", "return_offsets": True}
    assert context[0]["content"] == "program-specific instruction"


def test_unknown_variant_rejected():
    import pytest
    with pytest.raises(ValueError, match="unsupported"):
        variants.render_variant(None, [], {}, [], "unknown")
