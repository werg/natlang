export type TransformationRequest = {goal:string;allowedFiles:string[];preserves?:string[];changes?:string[];checks?:string[];evidence:unknown[]};
export type TransformationExplanation = {summary:string;changed:string[];preserves:string[]};
export type CounterexampleRequest={goal:string;contract:string;evidence:string;maxSuggestions:number};
export type CounterexampleSuggestions={inputs:unknown[][];reason:string};
