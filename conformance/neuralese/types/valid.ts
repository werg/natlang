// Programs the type rules accept (spec/SPEC.md, "Neuralese").
import { map, zip, ap, combine, empty, split, splitList, read, convert, type Neuralese } from 'natlang:neuralese';
import { grad, valueAndGrad, stopGradient, objectives, optimizers, type Loss } from 'natlang:learning';
import type { Bound, Context } from 'natlang:context';

type Ticket = { id: string; text: string };
type Label = 'urgent' | 'normal' | 'spam';
type Summary = { headline: string; points: string[] };

declare const ticket: Ticket;
declare const note: Neuralese<Summary>;
declare const rubric: Neuralese<string>;
declare const program: Bound<(s: Neuralese<string>) => Promise<Label>>;
declare const ctx: Context;

export async function examples() {
  // A literal's type comes from its annotation; the compiler emits the reference expression.
  const plan: Neuralese<Summary> = __neuralese.value('nz1_example');

  // Records may mix exact and soft fields.
  const brief: { summary: Neuralese<Summary>; path: string } = { summary: plan, path: 'a/b.md' };

  // read is the way out.
  const s: Summary = await read(brief.summary);
  const headline: string = s.headline;

  // Soft functions are callable with their declared parameters and result.
  const triage: Neuralese<(t: Ticket) => Promise<Label>> = __neuralese.value('nz1_triage');
  const label: Label = await triage(ticket);

  // Combinators.
  const titles: Neuralese<string> = await map(note, async (x: Summary) => x.headline);
  const pair: Neuralese<[Summary, string]> = await zip(note, rubric);
  const applied: Neuralese<Label> = await ap(triage, ticket);
  const merged: Neuralese<Summary> = await combine(note, plan, empty<Summary>());
  const parts: { headline: Neuralese<string>; points: Neuralese<string[]> } = await split(note);
  const items: Neuralese<string>[] = await splitList(parts.points);
  const moved: Neuralese<Summary, 'nd:natlang@2'> = await convert(note, 'nd:natlang@2');

  // Learning over context-held values.
  const loss = async (r: Neuralese<string>): Promise<Loss> =>
    objectives.crossEntropy(program.in(ctx.with({ rubric: r }))(r), 'urgent');
  const { loss: l, grad: g } = await valueAndGrad(loss, rubric);
  const g1 = await grad(loss, rubric);
  const opt = optimizers.adam({ lr: 1e-3 });
  const next = opt.step({ value: rubric, opt: opt.init(rubric) }, g);
  const frozen: Neuralese<string> = stopGradient(next.value);

  return { headline, label, titles, pair, applied, merged, items, moved, l, g1, frozen };
}
