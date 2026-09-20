"""Pending node kinds (SPEC 3, 4). Value nodes are plain Python data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import FoldT, IterateT, LambdaT, MapT, Type, TypeEnv


class _Missing:
    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self):
        return "MISSING"

    def __bool__(self):
        return False

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self


MISSING = _Missing()

UNREDUCED, RUNNING, QUIESCED, WAITING, DONE = "unreduced", "running", "quiesced", "waiting", "done"


@dataclass(eq=False)
class Pending:
    type: Type
    types: dict = field(default_factory=dict)  # name -> Type
    types_src: dict = field(default_factory=dict)  # name -> source text
    status: str = UNREDUCED
    note: str = ""
    steps: int = 0
    attempts: int = 0

    def env(self, outer: TypeEnv) -> TypeEnv:
        return outer.child(self.types)


@dataclass(eq=False)
class Lambda(Pending):
    type: LambdaT = None
    kind: str = "instructions"  # or "code"
    engine: str = "quickjs-isolated"  # crisp binding fixed when the function is loaded
    body: str = ""
    in_: dict = field(default_factory=dict)
    ret: Any = MISSING
    effects: list = field(default_factory=list)
    journal: list = field(default_factory=list)
    continuation_note: str = ""  # model-written reminder for a fresh conversation on this invocation
    original_body: Optional[str] = None  # body at first trigger, for reopen
    let: dict = field(default_factory=dict)        # typed locals (CODEBASES 2.1): name -> value or pending node
    let_types: dict = field(default_factory=dict)  # name -> Type, fixed by the write that created the local
    codebase: dict = field(default_factory=dict)   # name -> FunctionDef; immutable, shared by reference
    fn_name: str = ""                              # the function this lambda is an instance of, if any
    marks: dict = field(default_factory=dict)      # line number of the instructions -> "done" | "skipped"
    fn_copies: dict = field(default_factory=dict)  # local name -> FunctionDef it was copied from

    @property
    def is_crisp(self) -> bool:
        return self.kind == "code"


@dataclass(eq=False)
class MapNode(Pending):
    type: MapT = None
    over: Any = MISSING
    fn: Any = MISSING
    slots: Optional[list] = None  # None until expanded
    item_name: str = "item"       # the parameter of `fn` that receives the item


@dataclass(eq=False)
class FoldNode(Pending):
    type: FoldT = None
    over: Any = MISSING
    init: Any = MISSING
    step: Any = MISSING
    acc: Any = MISSING
    at: int = 0
    current: Any = None


@dataclass(eq=False)
class IterateNode(Pending):
    type: IterateT = None
    init: Any = MISSING
    step: Any = MISSING
    check: Any = MISSING
    max: Any = MISSING
    state: Any = MISSING
    iteration: int = 0
    recent: list = field(default_factory=list)
    seen_hashes: list = field(default_factory=list)
    current: Any = None
    state_name: str = "state"     # the parameter of `step` that receives the state
    check_name: str = ""          # "" : check(recent, iteration) -> LoopVerdict.  else: check(<name>: S) -> Bool


WRAPPERS = {"$lambda": Lambda, "$map": MapNode, "$fold": FoldNode, "$iterate": IterateNode}
WRAPPER_OF = {v: k for k, v in WRAPPERS.items()}


def is_pending(x) -> bool:
    return isinstance(x, Pending)
