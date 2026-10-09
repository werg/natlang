/** Point every [[from]] and [[from|shown]] link in a text at `to`. Plumbing: the replacement was decided elsewhere. */
export default function retarget(text: string, from: string, to: string): string {
  return text.split(`[[${from}]]`).join(`[[${to}]]`).split(`[[${from}|`).join(`[[${to}|`);
}
