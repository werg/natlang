/*---
engine: typescript-host
args:
  state: ConsoleState
  question: Text
  answer: EvidenceAnswer
returns: ConsoleState
---*/
return { questions: [...args.state.questions, args.question],
  answers: [...args.state.answers, args.answer], status: args.answer.status };
