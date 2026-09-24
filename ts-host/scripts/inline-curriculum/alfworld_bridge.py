"""ALFWorld bridge for the curriculum (same protocol as scienceworld_bridge.py).

serve:  python alfworld_bridge.py serve        (ALFWORLD_DATA must point at the downloaded data)
        ops: load (game: path relative to ALFWORLD_DATA), look, inventory, task, actions, act (command), score.
        A won game has score 100.
export: python alfworld_bridge.py export OUT.json --per-split 60
        Games from train, valid_seen, and valid_unseen with their task and the handcoded expert's full command list.
"""
import glob
import json
import os
import sys

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos

DATA = os.environ["ALFWORLD_DATA"]


def make(path, expert=False):
    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    wrappers = [AlfredDemangler(), AlfredInfos] + ([AlfredExpert(AlfredExpertType.HANDCODED)] if expert else [])
    env_id = textworld.gym.register_game(os.path.join(DATA, path), infos, max_episode_steps=1000, wrappers=wrappers)
    return textworld.gym.make(env_id)


def task_of(observation):
    return observation.split("Your task is to: ")[-1].strip()


def serve():
    env, state = None, {"won": False, "moves": 0, "obs": "", "admissible": []}
    for line in sys.stdin:
        request = json.loads(line)
        try:
            op = request["op"]
            if op == "load":
                env = make(request["task"])
                observation, info = env.reset()
                state.update(won=False, moves=0, obs=observation, admissible=info["admissible_commands"], task=task_of(observation))
                result = {"task": state["task"], "observation": observation}
            elif op == "task":
                result = state["task"]
            elif op in ("look", "inventory"):
                observation, _, _, info = env.step(op)
                state.update(admissible=info["admissible_commands"])
                result = observation
            elif op == "actions":
                result = {"commands": state["admissible"]}
            elif op == "act":
                observation, _, done, info = env.step(request["command"])
                state.update(won=state["won"] or bool(info["won"]), moves=state["moves"] + 1, admissible=info["admissible_commands"])
                result = {"observation": observation, "score": 100 if state["won"] else 0, "done": bool(done)}
            elif op == "score":
                result = {"score": 100 if state["won"] else 0, "done": state["won"], "moves": state["moves"]}
            else:
                raise ValueError(f"unknown op {op}")
            reply = {"id": request["id"], "result": result}
        except Exception as error:
            reply = {"id": request.get("id"), "error": str(error)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


def export(out, per_split):
    games = []
    for split in ("train", "valid_seen", "valid_unseen"):
        every = sorted(glob.glob(os.path.join(DATA, "json_2.1.1", split, "*", "*", "game.tw-pddl")))
        # Spread over the split: games are sorted by task type.
        paths = every[::max(1, len(every) // per_split)][:per_split]
        for full in paths:
            path = os.path.relpath(full, DATA)
            env = make(path, expert=True)
            observation, info = env.reset()
            plan = []
            for _ in range(80):
                if info["won"] or not info["extra.expert_plan"]:
                    break
                command = info["extra.expert_plan"][0]
                plan.append(command)
                observation, _, _, info = env.step(command)
            if info["won"]:
                games.append({"game": path, "split": split, "task": task_of(env.reset()[0]), "commands": plan})
            env.close()
    with open(out, "w") as handle:
        json.dump({"alfworld": "0.4.2", "games": games}, handle)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve()
    else:
        per_split = int(sys.argv[sys.argv.index("--per-split") + 1]) if "--per-split" in sys.argv else 60
        export(sys.argv[2], per_split)
