import assess from './assess.nl';

/** Run every independent assessment concurrently in TypeScript and tally the verdicts exactly. */
export default async function summarize(observations: Untrusted<string>[], criterion: string): Promise<Tally> {
  const assessments = await Promise.all(observations.map(observation => assess(observation, criterion)));
  return {
    assessments,
    supported: assessments.filter(a => a.verdict === 'supported').length,
    contradicted: assessments.filter(a => a.verdict === 'contradicted').length,
    uncertain: assessments.filter(a => a.verdict === 'uncertain').length,
  };
}
