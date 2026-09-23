"""ScienceWorld bridge for the curriculum.

serve:  python scienceworld_bridge.py serve
        JSON lines on stdin: {"id": 1, "op": "load", "task": "boil", "variation": 0, "simplifications": "easy"}
        ops: load, look, inventory, act (command), task, score, actions; each reply is one JSON line {"id", "result"} or {"id", "error"}.
export: python scienceworld_bridge.py export OUT.json --variations 3
        Task names, descriptions, and gold action paths for the first variations of every task.
"""
import json
import sys

from scienceworld import ScienceWorldEnv


def serve():
    env = ScienceWorldEnv("", envStepLimit=1000)
    state = {"score": 0, "done": False, "moves": 0}
    for line in sys.stdin:
        request = json.loads(line)
        try:
            op = request["op"]
            if op == "load":
                env.load(request["task"], request["variation"], request.get("simplifications", "easy"))
                observation, info = env.reset()
                state.update(score=info["score"], done=False, moves=0)
                result = {"task": env.get_task_description(), "observation": observation}
            elif op == "look":
                result = env.look()
            elif op == "inventory":
                result = env.inventory()
            elif op == "task":
                result = env.get_task_description()
            elif op == "score":
                result = {"score": state["score"], "done": state["done"], "moves": state["moves"]}
            elif op == "actions":
                result = {"templates": env.get_possible_actions(), "objects": env.get_possible_objects()}
            elif op == "act":
                observation, reward, done, info = env.step(request["command"])
                state.update(score=info["score"], done=bool(done), moves=info["moves"])
                result = {"observation": observation, "score": info["score"], "done": bool(done)}
            else:
                raise ValueError(f"unknown op {op}")
            reply = {"id": request["id"], "result": result}
        except Exception as error:  # the caller sees the error text
            reply = {"id": request.get("id"), "error": str(error)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


def export(out, variations):
    env = ScienceWorldEnv("", envStepLimit=1000)
    tasks = []
    for name in env.get_task_names():
        for variation in range(min(variations, env.get_max_variations(name))):
            env.load(name, variation, "easy", generateGoldPath=True)
            env.reset()
            tasks.append({"task": name, "variation": variation, "simplifications": "easy",
                          "description": env.get_task_description(), "gold": env.get_gold_action_sequence()})
    with open(out, "w") as handle:
        json.dump({"scienceworld": "1.2.2", "tasks": tasks}, handle)


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve()
    else:
        variations = int(sys.argv[sys.argv.index("--variations") + 1]) if "--variations" in sys.argv else 3
        export(sys.argv[2], variations)
