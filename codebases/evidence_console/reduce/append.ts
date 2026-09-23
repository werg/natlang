export default function append(state: ConsoleState, question: string, answer: EvidenceAnswer): ConsoleState {
return { questions: [...state.questions, question],
  answers: [...state.answers, answer], status: answer.status };
}
