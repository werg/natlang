---
args:
  request: CounterexampleRequest
returns: CounterexampleSuggestions
---
Suggest concrete deployment inputs that distinguish a plausible faulty implementation from the requested behavior. Use the public contract and training failure evidence in request. Return at most request.maxSuggestions argument lists with a short reason. Do not invent expected results: the independent author-supplied oracle will score these inputs. Do not request validation or test cases. Return inputs and reason; no source edits or evaluation authority belong to this reducer.
