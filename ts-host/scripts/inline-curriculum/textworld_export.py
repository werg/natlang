"""Generate TextWorld games with pinned seeds and export each as plain JSON for the curriculum's rule engine.

python textworld_export.py OUT_DIR --seeds 1-200

Each export holds the type hierarchy, every action rule (preconditions, postconditions, command template), the
world's facts, entity names and types, the quest objective (last action only), its winning commands, and the
win condition as facts. Games are generated with `tw-make custom`; the pinned TextWorld version is recorded.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import textworld


def predicate(p):
    return [p.name, [[v.name, v.type] for v in p.parameters]]


def proposition(p):
    return [p.name, [v.name for v in p.arguments]]


def export(game, seed, settings):
    kb = game.kb
    hierarchy = kb.logic.types
    types = {name: list(hierarchy.get(name).parents) for name in kb.types}
    rules = []
    for name, rule in sorted(kb.logic.rules.items()):
        rules.append({"name": name, "pre": [predicate(p) for p in rule.preconditions],
                      "post": [predicate(p) for p in rule.postconditions],
                      "command": kb.inform7_commands.get(name)})
    quest = game.quests[0]
    event = quest.win_events[0]
    return {
        "seed": seed, "settings": settings, "textworld": textworld.__version__,
        "types": types, "rules": rules,
        "facts": [proposition(f) for f in game.world.facts],
        "entities": {key: {"name": info.name, "type": info.type} for key, info in game.infos.items()},
        "objective": game.objective,
        "commands": list(quest.commands),
        "win": [proposition(p) for p in event.condition.preconditions],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seeds", default="1-50")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    os.makedirs(args.out, exist_ok=True)
    tw_make = os.path.join(os.path.dirname(sys.executable), "tw-make")
    for seed in range(lo, hi + 1):
        target = os.path.join(args.out, f"game-{seed}.json")
        if os.path.exists(target):
            continue
        rng = random.Random(seed)
        settings = {"world_size": rng.randint(4, 8), "nb_objects": rng.randint(8, 16), "quest_length": rng.randint(3, 8)}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "game.z8")
            subprocess.run([tw_make, "custom", "--world-size", str(settings["world_size"]), "--nb-objects", str(settings["nb_objects"]),
                            "--quest-length", str(settings["quest_length"]), "--only-last-action", "--seed", str(seed),
                            "--output", path, "--silent"], check=True)
            game = textworld.Game.load(path.replace(".z8", ".json"))
        with open(target, "w") as handle:
            json.dump(export(game, seed, settings), handle)
        print(target)


if __name__ == "__main__":
    main()
