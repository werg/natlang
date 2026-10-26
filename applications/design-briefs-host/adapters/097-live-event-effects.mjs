import { createHash } from 'node:crypto';

const APP = '097';
const TOPIC = Object.freeze({
  'reserve-event-resources': '097.live-event.resources.reserve',
  'request-role-acknowledgements': '097.live-event.roles.acknowledge',
  'publish-event-revision': '097.live-event.revision.publish',
});

function reject() {
  const error = new Error('invalid_effect_adapter');
  error.code = 'invalid_effect_adapter';
  throw error;
}

export default function liveEventEffectsAdapter({ result }) {
  if (!result || !Array.isArray(result.effects)) reject();
  return result.effects.map(effect => {
    if (!effect || typeof effect !== 'object' || Array.isArray(effect) ||
        !Object.hasOwn(TOPIC, effect.kind) || typeof effect.idempotencyKey !== 'string' || !effect.idempotencyKey)
      reject();
    const identity = JSON.stringify([APP, effect.kind, effect.idempotencyKey]);
    const key = `${APP}:${effect.kind}:${createHash('sha256').update(identity).digest('hex')}`;
    return { key, topic: TOPIC[effect.kind], payload: effect };
  });
}
