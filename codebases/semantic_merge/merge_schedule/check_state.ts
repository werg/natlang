export default function check_state(state: State): boolean {
const ids = state.events.map(event => event.id);
const utc = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && new Set(ids).size === ids.length &&
       state.events.every(event => utc.test(event.start) && utc.test(event.end) &&
                                Number.isFinite(Date.parse(event.start)) &&
                                Number.isFinite(Date.parse(event.end)) &&
                                Date.parse(event.start) < Date.parse(event.end));
}
