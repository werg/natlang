/**
 * The instructions of a `.nl` source as steps (crisp): its front matter, its body, the declaration a step judgment reads
 * (the front matter without `readout:`), and the body split into its numbered items or paragraphs. The shipped
 * `stepKind.nl` judges what each step asks for; the idiom reviewer and the program migrator read programs through these.
 */

/** What one instruction step asks the model to do (`stepKind.nl`). */
export type StepKind = 'judgment' | 'mechanism' | 'guard-for-type' | 'redundant';
/** One step with what it asks for. */
export type KindedStep = { text: string; kind: StepKind };
/** A `.nl` source read as instructions. */
export type Instructions = { front: string; body: string; declaration: string; steps: string[] };

/** The body's steps: its numbered items when it has them (text before the first one is a step of its own), else its
 * paragraphs. A fenced block stays inside the step it belongs to. */
export function stepsOf(body: string): string[] {
  const lines = body.replace(/\r\n/g, '\n').replace(/\s+$/, '').split('\n');
  const nextFence = (fence: string | undefined, line: string): string | undefined => {
    const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
    if (!marker) return fence;
    if (!fence) return marker[0]!.repeat(marker.length);
    return marker[0] === fence[0] && marker.length >= fence.length ? undefined : fence;
  };
  let fence: string | undefined;
  let numbered = false;
  for (const line of lines) {
    const updated = nextFence(fence, line);
    if (!fence && updated === undefined && /^\d+\.\s/.test(line)) { numbered = true; break; }
    fence = updated;
  }
  const steps: string[][] = [];
  fence = undefined;
  for (const line of lines) {
    const opens = !fence && (numbered ? /^\d+\.\s/.test(line) : line.trim() !== '' && (steps.length === 0 || steps.at(-1)!.at(-1)?.trim() === ''));
    if (opens || steps.length === 0) steps.push([]);
    steps.at(-1)!.push(line);
    fence = nextFence(fence, line);
  }
  return steps.map(step => step.join('\n').trim()).filter(Boolean);
}

/** A `.nl` source as instructions; throws when it does not start with front matter between `---` lines. */
export function instructionsOf(source: string): Instructions {
  const match = /^---\n([\s\S]*?)\n---\n?([\s\S]*)$/.exec(source.replace(/\r\n/g, '\n'));
  if (!match) throw new Error('a .nl file starts with front matter between --- lines');
  const front = match[1]!, body = match[2]!;
  const declaration = front.split('\n').filter(line => !/^readout:/.test(line)).join('\n').trim();
  return { front, body, declaration, steps: stepsOf(body) };
}
