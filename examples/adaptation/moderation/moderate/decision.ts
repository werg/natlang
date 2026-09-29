import { nl } from '@natlang/node';
export default async function decision(risk: 'safe' | 'unsafe'): Promise<boolean> {
  return await /* @natlangSite publishing */ nl<boolean>`Return true only when risk is safe.`();
}
