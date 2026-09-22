import { board_problem } from "./check_move/board_problem";
import { draw } from "./check_move/draw";
import { judge_move } from "./check_move/judge_move";
import { read_move } from "./check_move/read_move";
import { read_position } from "./check_move/read_position";
---
description: Is this move legal in this tic-tac-toe position, both described in prose? Exact rules, fuzzy reading.
args:
  position: Text
  move: Text
  player: Mark
returns: Verdict
types:
  Mark: '"X" | "O"'
  Cell: '"X" | "O" | "empty"'
  Verdict: '{ legal: Bool, reason: Text, wins: Bool, board: Text }'
---
function check_move(position, move, player) -> Verdict

  cells   = read_position(position)               # nine cells, row by row, from the prose
  problem = board_problem(cells, player)           # exact: "" if the position is possible and it is this player's turn
  if problem is not "":
      return { legal: false, reason: problem, wins: false, board: draw(cells) }
  target  = read_move(move)                        # which cell, 1 to 9; if the move does not name one cell, that is a blocker
  return judge_move(cells, target, player)         # exact: occupied? out of range? does it win?
