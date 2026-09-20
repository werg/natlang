import { define, initial, COMMON, action as a, field as f, panel as p, form, table, find, text, number, requireValue as check, revised, uid, unsupported, rng } from '../shared/domain.mjs';
export const economy = define({ id: 'economy', project: 'P08', title: 'Market day', subtitle: 'Every exchange tells a story', category: 'Play', icon: '◉', color: '#e4bc72', panelIds: ['market', 'trade', 'ledger'],
    stateType: `export type Merchant = { id: Text; name: Text; cash: Num; apples: Num; price: Num }; export type Trade = { id: Text; buyer: Text; seller: Text; quantity: Num; total: Num }; export type State = { ${COMMON} merchants: Merchant[]; trades: Trade[]; tick: Num };`,
    initial: () => initial({ merchants: [{ id: 'you', name: 'Your stall', cash: 100, apples: 4, price: 5 }, { id: 'mira', name: 'Mira’s orchard', cash: 80, apples: 20, price: 4 }, { id: 'sol', name: 'Sol’s kitchen', cash: 120, apples: 2, price: 7 }], trades: [], tick: 0 }),
    instructions: 'Operate a small closed economy. buy moves amount apples from target seller to secondary buyer, transferring cash at seller price. price sets target merchant offer to amount. When asked to simulate an actor, inspect stocks/cash/offers and choose one sensible trade or price adjustment; explain via the view. Cash and apples must be conserved. No invented production, credit, or successful trades.',
    apply(s, d) { const m = find(s.merchants, d.target); if (d.action === 'price')
        { check(Number.isSafeInteger(d.amount),'Prices use whole coins'); m.price = number(d.amount, 'Price', 1, 1000); }
    else if (d.action === 'buy') {
        const buyer = find(s.merchants, d.secondary);
        check(buyer.id !== m.id, 'Choose different buyer and seller');
        const qty = number(d.amount, 'Quantity', 1, 1000);
        check(Number.isInteger(qty), 'Use whole apples');
        const total = qty * m.price;
        check(m.apples >= qty, 'Seller has insufficient stock');
        check(buyer.cash >= total, 'Buyer has insufficient cash');
        m.apples -= qty;
        buyer.apples += qty;
        m.cash += total;
        buyer.cash -= total;
        s.trades.push({ id: uid(s.trades), buyer: buyer.id, seller: m.id, quantity: qty, total });
        s.tick++;
    }
    else
        unsupported(d.action); return revised(s, 'The market has a new balance.'); },
    panels: s => [p('market', 'The morning market', 'market', { merchants: s.merchants, metrics: [['Coins in circulation', s.merchants.reduce((n, m) => n + m.cash, 0)], ['Apples in the market', s.merchants.reduce((n, m) => n + m.apples, 0)], ['Exchanges', s.tick]] }), form('trade', 'Make an exchange', [f('target', 'Seller', 'mira', 'select', s.merchants.map(m => m.id)), f('secondary', 'Buyer', 'you', 'select', s.merchants.map(m => m.id)), f('amount', 'Apples', 2, 'number')], [a('Trade apples', 'buy', {}, 'primary')], { extraForms: [{ title: 'Change an offer', fields: [f('target', 'Merchant', 'you', 'select', s.merchants.map(m => m.id)), f('amount', 'Price per apple', 5, 'number')], actions: [a('Set price', 'price')] }] }), table('ledger', 'Every coin accounted for', s.trades, ['buyer', 'seller', 'quantity', 'total'])], smoke: { action: 'buy', target: 'mira', secondary: 'you', amount: 2 } });
export const combat = define({ id: 'combat', project: 'P08', title: 'Sparring grounds', subtitle: 'Read the moment. Choose your move.', category: 'Play', icon: '⚔', color: '#db9d9e', panelIds: ['arena', 'moves', 'rounds'],
    stateType: `export type Fighter = { id: Text; hp: Num; energy: Num; stance: Text }; export type Round = { id: Text; player: Text; rival: Text; detail: Text }; export type State = { ${COMMON} fighters: Fighter[]; rounds: Round[]; seed: Num };`,
    initial: () => initial({ fighters: [{ id: 'you', hp: 100, energy: 10, stance: 'ready' }, { id: 'rival', hp: 100, energy: 10, stance: 'ready' }], rounds: [], seed: 17 }),
    instructions: 'Resolve simultaneous combat decisions. round sets player move in text and rival move in secondary, one of strike, guard, recover. Select a rival move semantically from the public state; never claim to observe a hidden player intent beyond supplied UI controls. Exact host physics handles energy, damage and simultaneous resolution. reset starts a new match. No attacks after the match ends.',
    apply(s, d) { if (d.action === 'reset')
        return revised(combat.initial(), 'A fresh match.'); check(d.action === 'round', 'Unknown combat action'); const moves = [d.text, d.secondary], fighters = s.fighters; check(fighters.every(f => f.hp > 0), 'Match finished; start a new match'); moves.forEach((move, i) => { check(['strike', 'guard', 'recover'].includes(move), 'Unknown move'); check(move !== 'strike' || fighters[i].energy >= 3, 'Not enough energy to strike'); }); const damage = moves.map((move, i) => move === 'strike' ? (moves[1 - i] === 'guard' ? 3 : 16) : 0); fighters.forEach((f, i) => { f.hp = Math.max(0, f.hp - damage[1 - i]); f.energy = Math.min(10, f.energy + (moves[i] === 'recover' ? 4 : moves[i] === 'strike' ? -3 : 1)); f.stance = moves[i]; }); s.rounds.push({ id: uid(s.rounds), player: moves[0], rival: moves[1], detail: `You took ${damage[1]}; rival took ${damage[0]}.` }); return revised(s, fighters.some(f => f.hp === 0) ? 'The match is complete.' : 'Take a breath. Choose the next exchange.'); },
    panels: s => [p('arena', 'The circle', 'arena', { fighters: s.fighters }), form('moves', 'Plan the exchange', [f('text', 'Your move', 'strike', 'select', ['strike', 'guard', 'recover']), f('secondary', 'Rival move · explicit control', 'guard', 'select', ['strike', 'guard', 'recover'])], [a('Resolve round', 'round', {}, 'primary'), a('New match', 'reset')], { footer: 'For a natlang-controlled rival, describe your move in the command bar and ask it to choose the response.' }), table('rounds', 'Round journal', s.rounds, ['id', 'player', 'rival', 'detail'])], smoke: { action: 'round', text: 'strike', secondary: 'guard' } });
export const npc = define({ id: 'npc', project: 'P08', title: 'The lantern inn', subtitle: 'A conversation with a memory', category: 'Play', icon: '☷', color: '#c9b18e', panelIds: ['scene', 'conversation', 'memory'],
    stateType: `export type Memory = { id: Text; text: Text }; export type Line = { id: Text; speaker: Text; text: Text; evidence: Text[] }; export type State = { ${COMMON} memories: Memory[]; lines: Line[]; trust: Num };`,
    initial: () => initial({ memories: [{ id: 'bridge', text: 'The east bridge was washed out last night.' }, { id: 'ferry', text: 'Tomas runs a ferry at dawn, but needs help carrying supplies.' }, { id: 'keeper', text: 'I keep the Lantern Inn. I value kindness and dislike empty promises.' }], lines: [{ id: 'item-1', speaker: 'keeper', text: 'Come in from the rain. Where are you headed?', evidence: ['keeper'] }], trust: 0 }),
    instructions: 'Play the innkeeper with evidence-linked memory. say appends user text. reply speaks as the keeper using text, cites memory IDs in ids, and adjusts trust by amount between -1 and 1. Remember only established facts; do not invent past events. remember stores a confirmed fact in text. When asked to respond, ground specific factual statements in known memories and convey uncertainty when needed.',
    apply(s, d) { if (d.action === 'say')
        s.lines.push({ id: uid(s.lines), speaker: 'you', text: text(d.text), evidence: [] });
    else if (d.action === 'reply') {
        (d.ids ?? []).forEach(id => find(s.memories, id));
        s.lines.push({ id: uid(s.lines), speaker: 'keeper', text: text(d.text), evidence: d.ids ?? [] });
        s.trust = Math.max(-5, Math.min(5, s.trust + number(d.amount ?? 0, 'Trust change', -1, 1)));
    }
    else if (d.action === 'remember')
        s.memories.push({ id: uid(s.memories), text: text(d.text) });
    else
        unsupported(d.action); return revised(s, 'The conversation continues.'); },
    panels: s => [p('scene', 'Shelter from the rain', 'scene', { body: 'The Lantern Inn', detail: 'Rain on the windows. A low fire. Someone who remembers.', metrics: [['Trust', s.trust], ['Memories', s.memories.length]] }), form('conversation', 'By the fireside', [f('text', 'What do you say?', 'I need to reach the eastern village.', 'textarea')], [a('Speak', 'say')], { chat: s.lines, footer: 'After speaking, ask natlang to reply as the innkeeper. Fixture mode does not generate dialogue.' }), table('memory', 'What the keeper knows', s.memories, ['id', 'text'])], smoke: { action: 'say', text: 'Can you tell me about the ferry?' } });
export const spells = define({ id: 'spells', project: 'P09', title: 'Spellweaver', subtitle: 'Words become a little bit of magic', category: 'Play', icon: '✧', color: '#b8a5ef', panelIds: ['arena', 'spell', 'grimoire'],
    stateType: `export type Spell = { id: Text; words: Text; element: Text; power: Num; cost: Num }; export type State = { ${COMMON} mana: Num; targetHp: Num; ward: Text; spells: Spell[]; last: Text };`,
    initial: () => initial({ mana: 40, targetHp: 100, ward: 'ice', spells: [], last: 'The practice sentinel waits.' }),
    instructions: 'Translate spells into exact actions. cast uses words in text, element in target (fire, ice, wind, light), power amount (1 to 10). Cost is twice power; host checks mana and resolves damage against the sentinel ward. Use creative wording to select element/power, but never invent mechanics. rest refills mana and resets sentinel. The typed action is the proposed compilation; exact physics is host code.',
    apply(s, d) { if (d.action === 'rest') {
        s.mana = 40;
        s.targetHp = 100;
        s.last = 'The circle is restored.';
    }
    else if (d.action === 'cast') {
        check(s.targetHp > 0, 'The sentinel is already defeated');
        check(['fire', 'ice', 'wind', 'light'].includes(d.target), 'Unknown element');
        const power = number(d.amount, 'Power', 1, 10);
        check(Number.isInteger(power), 'Power must be an integer');
        const cost = power * 2;
        check(s.mana >= cost, 'Not enough mana');
        s.mana -= cost;
        const damage = power * (d.target === 'fire' ? 3 : d.target === 'ice' ? 1 : 2);
        s.targetHp = Math.max(0, s.targetHp - damage);
        s.last = `${d.target} dealt ${damage} damage for ${cost} mana.`;
        s.spells.push({ id: uid(s.spells), words: text(d.text), element: d.target, power, cost });
    }
    else
        unsupported(d.action); return revised(s, s.last); },
    panels: s => [p('arena', 'The practice circle', 'spell-arena', { mana: s.mana, hp: s.targetHp, last: s.last }), form('spell', 'Write a spell', [f('text', 'Incantation', 'Embers of morning, thaw this frozen heart.', 'textarea'), f('target', 'Element · explicit control', 'fire', 'select', ['fire', 'ice', 'wind', 'light']), f('amount', 'Power', 4, 'number')], [a('Cast spell', 'cast', {}, 'primary'), a('Restore circle', 'rest')], { footer: 'Use the command bar to let natlang compile your own incantation into an action.' }), table('grimoire', 'Your grimoire', s.spells, ['words', 'element', 'power', 'cost'])], smoke: { action: 'cast', text: 'A small warm spark', target: 'fire', amount: 3 } });
export const experiments = define({ id: 'experiments', project: 'P16', title: 'Possibility lab', subtitle: 'Same world. Different choices.', category: 'Explore', icon: '⎇', color: '#92c9ce', panelIds: ['design', 'chart', 'trials'],
    stateType: `export type Trial = { id: Text; policy: Text; seed: Num; reward: Num; steps: Num }; export type State = { ${COMMON} trials: Trial[]; hypothesis: Text };`,
    initial: () => initial({ trials: [], hypothesis: 'Does a cautious inventory policy waste less while serving demand?' }),
    algorithm: 'For compare, run the cautious policy and then the generous policy using the same supplied seed and step count. Inspect both results and explain the comparison. For further experiments, select seeds and horizons deliberately and call run for each trial; do not hide the experimental design in a host loop.',
    instructions: 'Design matched seeded inventory experiments. hypothesis stores text. run uses target policy cautious or generous, amount seed (nonnegative integer), count steps (positive integer). The natlang caller compares policies by calling run repeatedly with matched seed and horizon. Exact simulation samples demand 0..9 per step; cautious stocks 4, generous stocks 8; reward is 3*units sold - stock - 2*unserved demand. Explain tradeoffs only from recorded trials. No arbitrary episode limit is imposed.',
    async apply(s, d, host) { if (d.action === 'hypothesis')
        s.hypothesis = text(d.text);
    else if (d.action === 'run') {
        const seed = number(d.amount, 'Seed', 0, 4294967295), steps = number(d.count, 'Steps', 1);
        check(Number.isSafeInteger(seed) && Number.isSafeInteger(steps), 'Use whole seed and step values');
        const policy=d.target;
        check(['cautious','generous'].includes(policy),'Unknown policy');
        const reward=await host.trial({seed,steps,policy});
        s.trials.push({id:uid(s.trials),policy,seed,reward,steps});
    }
    else
        unsupported(d.action); return revised(s, 'Experiment recorded with its seed and horizon.'); },
    panels: s => [form('design', 'Ask a testable question', [f('text', 'Hypothesis', s.hypothesis, 'textarea')], [a('Save hypothesis', 'hypothesis')], { extraForms: [{ title: 'Matched comparison', fields: [f('amount', 'Random seed', 17, 'number'), f('count', 'Simulation steps', 100, 'number')], actions: [a('Run both policies', 'compare', {}, 'primary')] }] }), p('chart', 'What happened', 'chart', { rows: s.trials.map(t => ({ label: `${t.policy} · seed ${t.seed}`, value: t.reward })) }), table('trials', 'Reproducible trials', s.trials, ['policy', 'seed', 'steps', 'reward'])], smoke: { action: 'run', target: 'cautious', amount: 17, count: 100 } });
export default [economy, combat, npc, spells, experiments];

/** Exact baseline world mechanics, executed in a worker by the browser host. */
export function simulateInventory({seed,steps,policy}) {
 const random=rng(seed);let reward=0;
 for(let i=0;i<steps;i++){const demand=Math.floor(random()*10),stock=policy==='cautious'?4:8;reward+=3*Math.min(stock,demand)-stock-2*Math.max(0,demand-stock);}
 check(Number.isSafeInteger(reward),'Trial reward exceeds exact numeric representation');return reward;
}
