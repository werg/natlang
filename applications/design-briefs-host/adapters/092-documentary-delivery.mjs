import { createHash } from 'node:crypto';

const APP = '092';
const TOPIC = Object.freeze({
  'deliver-documentary-package': '092.documentary.package.deliver',
  'read-documentary-delivery-status': '092.documentary.delivery.status.read',
});

function reject() {
  const error = new Error('invalid_effect_adapter');
  error.code = 'invalid_effect_adapter';
  throw error;
}

export default function documentaryDeliveryAdapter({ result }) {
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
