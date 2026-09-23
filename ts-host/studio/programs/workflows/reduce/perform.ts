import { studio } from 'natlang:services';
export default async function perform(state: State, event: UiEvent, decision: Decision): Promise<Step> {
  return await studio.apply(state, event, decision);
}
