import { nl } from '@natlang/node';
import classify from './classify.nl';
export async function triage(message: string): Promise<string> {
  const label = await classify(message);
  return await /* @natlangSite normalize */ nl<string>`Return label unchanged. The original message is ${message}.`();
}
