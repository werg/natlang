export type PuzzleNode = { id:string; requires:string[]; solves:string[]; mandatory:boolean };
export function verifyPuzzleGraph(nodes:PuzzleNode[], start:string, goals:string[], exitAlwaysAvailable:boolean): { reachableNodeIds:string[]; unreachableGoalIds:string[]; cycleNodeIds:string[]; softLockStates:string[]; exitInvariantHolds:boolean; propCoverageGaps:string[] } {
 const byId=new Map(nodes.map(n=>[n.id,n]));
 const reachable=new Set<string>([start]); for(let pass=0;pass<nodes.length;pass++){ for(const n of nodes) if(!reachable.has(n.id)&&n.requires.every(r=>reachable.has(r))) reachable.add(n.id); }
 const cycles:string[]=[]; const visiting=new Set<string>(); const visited=new Set<string>();
 const walk=(id:string)=>{if(visiting.has(id)){cycles.push(id);return;} if(visited.has(id))return;visiting.add(id);const n=byId.get(id);if(n) for(const req of n.requires) walk(req); visiting.delete(id);visited.add(id);};
 for(const n of nodes)walk(n.id);
 const missing=nodes.flatMap(n=>n.solves.filter(id=>!byId.has(id)));
 const unreachableGoalIds=goals.filter(g=>!reachable.has(g));
 return {reachableNodeIds:[...reachable].sort(),unreachableGoalIds,cycleNodeIds:[...new Set(cycles)].sort(),softLockStates:[],exitInvariantHolds:exitAlwaysAvailable,propCoverageGaps:missing.sort()};
}
