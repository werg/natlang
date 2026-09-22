import yaml

from natlang.explicit_surface import ExplicitToolSurface
from natlang.decoder import ChatTurn
from natlang.native import call_grammar
from natlang.runtime import Runtime, Session
from natlang.surface_projection import project_actions_v4
from natlang.tool_agent import ToolAgent
from natlang.types import TypeEnv
from natlang.values import coerce, load_program


def _session(document):
    root = load_program(document["program"])
    env = root.env(TypeEnv())
    for name, value in document.get("inputs", {}).items():
        root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    return Session(Runtime(None), root, TypeEnv())


def test_explicit_surface_splits_write_and_call_modes():
    document = yaml.safe_load("""
program:
  $lambda:
    type: 'Lambda<{ lines: Line[] }, Num>'
    types: { Line: '{ qty: Num, price: Num }' }
    instructions: Add all line totals.
    codebase:
      add:
        args: { acc: Num, item: Line }
        returns: Num
        code: return args.acc + args.item.qty * args.item.price
      enough:
        args: { state: Num }
        returns: Bool
        code: return args.state > 20
inputs:
  lines: [{ qty: 2, price: 4 }, { qty: 3, price: 5 }]
""")
    session = _session(document)
    surface = ExplicitToolSurface()
    tools = surface.tools(session)
    names = [tool["function"]["name"] for tool in tools]
    assert "call" not in names and "write" not in names
    assert {"write_value", "copy_function", "run_function", "for_each", "fold", "repeat"} <= set(names)
    fold_tool = next(tool for tool in tools if tool["function"]["name"] == "fold")
    grammar = call_grammar([fold_tool])
    assert "item_param" not in grammar and "accumulator_param" not in grammar

    assert session.apply("write_value", {"destination": "let/zero", "type": "Num", "value": 0}).kind == "ok"
    result = session.apply("fold", {"function": "add", "save_as": "return", "items": "args/lines",
                                    "initial": "let/zero"})
    assert result.kind == "done" and session.lam.ret == 23


def test_exact_edit_is_content_constrained_and_fuzzy_edit_is_unambiguous():
    document = yaml.safe_load("""
program:
  $lambda:
    type: 'Lambda<{}, Text>'
    instructions: Adapt greet, then use it.
    codebase:
      greet:
        args: { name: Text }
        returns: Text
        instructions: |
          Say hello to `args/name`.
          Keep the reply short.
""")
    session = _session(document)
    surface = ExplicitToolSurface()
    assert session.apply("copy_function", {"function": "greet", "save_as": "let/friendly"}).kind == "ok"
    edit = next(tool for tool in surface.tools(session) if tool["function"]["name"] == "edit_text")
    alternatives = edit["function"]["parameters"]["x-natlang-alternatives"]
    exact = [alt for alt in alternatives if "fuzzy" not in alt]
    assert any(alt["path"] == {"const": "let/friendly/instructions"} and
               "Say hello to `args/name`.\n" in alt["find"]["enum"] for alt in exact)

    result = session.apply("edit_text", {"path": "let/friendly/instructions",
                                          "find": "Say hullo to args/name",
                                          "replace_with": "Welcome `args/name`.\n", "fuzzy": True})
    assert result.kind == "ok"
    assert session.lam.let["friendly"].body.startswith("Welcome `args/name`.")


def test_parallel_batch_waits_for_an_explicit_workspace_dependency():
    root = load_program(yaml.safe_load("""
$lambda:
  type: 'Lambda<{}, Text>'
  instructions: Read the event, then transition it.
  codebase:
    read_event:
      args: {}
      returns: Text
      code: return "created"
    transition:
      args: { kind: Text }
      returns: Text
      code: return args.kind + "!"
"""))

    class Scripted:
        def __init__(self):
            self.turns = [
                ChatTurn([("run_function", {"function": "transition", "save_as": "return",
                                              "inputs": ["let/kind"]}),
                          ("run_function", {"function": "read_event", "save_as": "let/kind"})]),
                ChatTurn([], ""),
            ]

        def chat(self, messages, tools, **kwargs):
            return self.turns.pop(0)

    outcome, value = Runtime(lambda lam: ToolAgent(
        Scripted(), surface=ExplicitToolSurface(marks=False))).run_root(root)
    assert outcome.kind == "done" and value == "created!"


def test_historical_named_call_projects_to_positional_paths_and_explicit_literal():
    document = yaml.safe_load("""
program:
  $lambda:
    type: 'Lambda<{ text: Text }, Text>'
    instructions: Replace a word.
    codebase:
      replace:
        args: { text: Text, old: Text, new: Text }
        returns: Text
        code: return args.text.replace(args.old, args.new)
inputs: { text: hello }
""")
    tools = ExplicitToolSurface().tools(_session(document))
    actions = project_actions_v4("call", {
        "function": "replace", "to": "return", "inputs": {"text": "args/text"},
        "values": {"old": "hell", "new": "y"}, "done": 1}, offered_tools=tools)
    assert actions == [
        ("write_value", {"destination": "let/migrated_old", "type": "Text", "value": "hell"}),
        ("write_value", {"destination": "let/migrated_new", "type": "Text", "value": "y"}),
        ("run_function", {"function": "replace",
                          "inputs": ["args/text", "let/migrated_old", "let/migrated_new"],
                          "save_as": "return"}),
        ("mark_lines", {"start": 1}),
    ]
