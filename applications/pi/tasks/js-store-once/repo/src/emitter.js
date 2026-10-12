/** Minimal event emitter: listeners run in registration order. */
export class Emitter {
  constructor() {
    this.listeners = new Map();
  }

  /** Register fn for event; returns a function that removes it. */
  on(event, fn) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event).push(fn);
    return () => this.off(event, fn);
  }

  /** Register fn for the next emit of event only. */
  once(event, fn) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      fn(...args);
    };
    return this.on(event, wrapper);
  }

  /** Remove fn (registered with on or once) from event. */
  off(event, fn) {
    const list = this.listeners.get(event);
    if (!list) return;
    const index = list.indexOf(fn);
    if (index !== -1) list.splice(index, 1);
  }

  emit(event, ...args) {
    const list = this.listeners.get(event) || [];
    for (let i = 0; i < list.length; i++) list[i](...args);
  }

  count(event) {
    return (this.listeners.get(event) || []).length;
  }
}
