export default function check_state(state: State): boolean {
return Number.isSafeInteger(state.revision) && state.revision >= 0 &&
       Object.keys(state.fields).every(key => key.trim().length > 0);
}
