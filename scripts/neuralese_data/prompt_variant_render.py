"""Render a source-bound prompt variant without changing the offered tool contract.

This deliberately does not grant source admission: callers must authenticate the
original trajectory and apply the complete held-world protection before publishing.
"""
from natlang_neuralese.data.prompt_rendering import apply_prompt_rendering
from natlang_neuralese.data.text_corpus import native_gold_packet


def render_variant(tokenizer, context, target, tools, mode, **packet_options):
    if mode not in ("full", "program-header"):
        raise ValueError("unsupported prompt variant")
    turns, offered = apply_prompt_rendering(context, tools, mode, keep_supplied_tools=True)
    if offered != tools:
        raise ValueError("prompt variant changed the offered tool contract")
    # The header is already rendered. Rendering as full preserves supplied tools
    # and their typed schemas both in the prompt and in target serialization.
    return native_gold_packet(tokenizer, turns, target, offered,
                              prompt_rendering="full", **packet_options)
