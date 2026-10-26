import { createHash } from 'node:crypto';

const APP = '098';
const TOPIC = Object.freeze({
  'apply-player-actions': '098.persistent-world.actions.apply',
  'read-action-status': '098.persistent-world.action.status.read',
});

function reject() {
  const error = new Error('invalid_effect_adapter');
  error.code = 'invalid_effect_adapter';
  throw error;
}

export default function persistentWorldActionsAdapter({ result }) {
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
