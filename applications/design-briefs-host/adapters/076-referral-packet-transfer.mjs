import { createHash } from 'node:crypto';

const APP = '076';
const TOPIC = Object.freeze({
  'send-referral-packet': '076.referral.packet.send',
  'read-recipient-status': '076.referral.status.read',
});

function reject() {
  const error = new Error('invalid_effect_adapter');
  error.code = 'invalid_effect_adapter';
  throw error;
}

export default function referralPacketTransferAdapter({ result }) {
  if (!result || !Array.isArray(result.effects)) reject();
  return result.effects.map(effect => {
    if (!effect || typeof effect !== 'object' || Array.isArray(effect) ||
        !Object.hasOwn(TOPIC, effect.kind) || typeof effect.effectId !== 'string' || !effect.effectId)
      reject();
    const identity = JSON.stringify([APP, effect.kind, effect.effectId]);
    const key = `${APP}:${effect.kind}:${createHash('sha256').update(identity).digest('hex')}`;
    return { key, topic: TOPIC[effect.kind], payload: effect };
  });
}
