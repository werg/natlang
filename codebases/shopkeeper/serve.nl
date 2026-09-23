---
description: A shopkeeper handles one customer message - understand it, act only
  within what is legal, reply, update the shop.
args:
  acc: Shop
  item: Message
returns: Shop
effects:
  - out.emit
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
