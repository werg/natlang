import { Emitter } from './emitter.js';

/** A value holder that tells subscribers about every change. */
export function createStore(initial) {
  let state = initial;
  const events = new Emitter();
  return {
    get: () => state,
    set(next) {
      const previous = state;
      state = next;
      events.emit('change', next, previous);
    },
    subscribe: (fn) => events.on('change', fn),
    subscribeOnce: (fn) => events.once('change', fn),
    unsubscribe: (fn) => events.off('change', fn),
    subscribers: () => events.count('change'),
  };
}
