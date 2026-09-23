export default function submit(actor: string, tick: number, intent: TradeIntent): TradeReceipt {
return host.economy.submit(actor, tick, intent);
}
