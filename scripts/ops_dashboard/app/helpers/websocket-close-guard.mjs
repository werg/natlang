// Node's (and every browser's) WebSocket.close(code) accepts only 1000 and 3000-4999 from a client; any other code throws
// InvalidAccessError ("invalid code"). Tendriller's invitation listener (@tendriller/core invitations.mjs, rc.17) closes its
// socket with 1011 from the socket's own `error` event and with 1008 on a bad frame, so a host that refuses the connection
// (e.g. HTTP 404 "No such tendril") throws inside an event listener, which Node turns into an uncaught exception and the
// participant dies (ops-peer-16, 2026-11-22 05:43). Until that is fixed upstream, a close with a code a client may not send
// is sent as a normal close (1000) with the same reason. A no-op for every valid code.
const valid = code => code === undefined || code === 1000 || (Number.isInteger(code) && code >= 3000 && code <= 4999);
const native = globalThis.WebSocket?.prototype?.close;
if (typeof native === 'function' && !native.guarded) {
  const guarded = function close(code, reason) {
    return valid(code) ? native.call(this, code, reason) : native.call(this, 1000, reason);
  };
  guarded.guarded = true;
  globalThis.WebSocket.prototype.close = guarded;
}
