export default async function apply(state: State, event: UiEvent, decision: Decision): Promise<Step> {
return await host.studio.apply(state, event, decision);
}
