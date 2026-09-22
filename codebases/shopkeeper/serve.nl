import { apply_action } from "./serve/apply_action";
import { checked_action } from "./serve/checked_action";
import { choose_action } from "./serve/choose_action";
import { emit_reply } from "./serve/emit_reply";
import { goods_of } from "./serve/goods_of";
import { legal_actions } from "./serve/legal_actions";
import { read_intent } from "./serve/read_intent";
import { say } from "./serve/say";
---
description: A shopkeeper handles one customer message - understand it, act only within what is legal, reply, update the shop.
args:
  acc: Shop
  item: Message
returns: Shop
effects: [out.emit]
types:
  Message: '{ from: string, text: string }'
  Shop: '{ stock: Record<string, number>, prices: Record<string, number>, coins: number, persona: string, ledger: string[] }'
  Intent: '{ kind: "buy" | "ask_price" | "haggle" | "chat", good: string, qty: number, offer: number }'
  Action: '{ code: string, good: string, qty: number, price: number }'
---
function serve(acc, item) -> Shop

  goods   = goods_of(acc)                                  # exact: what the shop sells
  intent  = read_intent(item.text, goods)                  # what does the customer want?
  legal   = legal_actions(acc, intent)                     # exact: every action the rules allow right now
  wish    = choose_action(acc.persona, intent, legal)      # the shopkeeper's choice, by code, from `legal`
  action  = checked_action(wish, legal)                    # exact: an illegal or unknown choice becomes "decline"
  line    = say(acc.persona, item.text, action)            # the reply, in character
  emit_reply(item.from, line, action)                      # effect
  return apply_action(acc, action, item.from)              # exact: stock, coins, ledger
