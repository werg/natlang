"""Construct fresh service worlds; only sandbox teachers generate model trajectories."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds

REVISION='authored-service-recovery-20261016/1'
DOMAINS={'inventory':('stock','available'),'finance':('ledger','approved'),
    'builds':('builds','passed'),'telemetry':('metrics','critical'),
    'support':('tickets','urgent'),'shipping':('shipments','ready'),
    'packages':('catalog','stable'),'research':('evidence','confirmed')}
VARIANTS=('namespace-reuse','api-recovery','typed-argument-recovery','readback-recovery')
TYPES='export type Row = { id:string; category:string; value:number; active:boolean };\nexport type Policy = { category:string; minimum:number };\nexport type Summary = { ids:string[]; total:number; count:number };\n'


def worlds(shapes=16, *, revision=REVISION, domains=DOMAINS):
    for domain,(namespace,category) in domains.items():
        for shape in range(shapes):
            rng=random.Random(f'{revision}:{domain}:{shape}')
            rows=[{'id':f'{domain}-{shape:03d}-{i:03d}',
                'category':rng.choice([category,'pending','excluded']),
                'value':rng.randint(2,800),'active':rng.random()>.2} for i in range(13+shape%9)]
            policy={'category':category,'minimum':70+shape*9};page_size=4+shape%3
            kept=[row for row in rows if row['category']==category and row['active'] and row['value']>=policy['minimum']]
            expected={'ids':sorted(row['id'] for row in kept),'total':sum(row['value'] for row in kept),'count':len(kept)}
            group=f'{revision}:{domain}:world-{shape}'
            predicate='record.category === policy.category && record.active && record.value >= policy.minimum'
            children=[{'match':['select_record(',row['id']], 'calls':[
                ['eval',{'code':'return '+predicate+';'}],
                ['return_result',{'status':'success','value':row in kept}]]} for row in rows]
            reference=(f'const pages = await Promise.all(Array.from({{length: await {namespace}.page_count()}}, (_, index) => {namespace}.page(index + 1))); '
                'const rows = pages.flat(); '
                'const decisions = await Promise.all(rows.map(async record => ({record,keep:await select_record(record,policy)}))); '
                'const selected = decisions.filter(item=>item.keep).map(item=>item.record); '
                'const result = {ids:selected.map(item=>item.id).sort(),total:selected.reduce((sum,item)=>sum+item.value,0),count:selected.length}; '
                "await folder.file('summary.json').writeText(JSON.stringify(result)+'\\n'); "
                "const saved = JSON.parse(await folder.file('summary.json').readText()); return saved;")
            service=(TYPES+f'const ROWS:Row[] = {json.dumps(rows,separators=(",",":"))};\n'
                f'/** Return page n (1-based) of {domain} records; an empty page follows the last. */\n'
                f'export function page(n:number):Row[] {{return ROWS.slice((n-1)*{page_size},n*{page_size});}}\n'
                f'/** Exact number of {domain} pages. */\nexport function page_count():number {{return {((len(rows)+page_size-1)//page_size)};}}\n')
            seeds={
                'namespace-reuse':f'const {namespace} = {namespace}.page(1); return {namespace};',
                'api-recovery':f'return {namespace}.get_all_pages();',
                'typed-argument-recovery':'return await select_record({id:"seed",category:policy.category,value:"unavailable",active:true},policy);',
                'readback-recovery':f"return JSON.parse(await folder.file('missing-{domain}-{shape}.json').readText());"}
            for variant in VARIANTS:
                identifier=f'{group}:{variant}'
                yield {'version':'natlang.program/2','id':identifier,'kind':'lambda_source',
                    'family':'curriculum_fresh_service_recovery','source':'natlang-inline-curriculum',
                    'split':'train','source_ids':[group],'source_groups':[group],
                    'source_revisions':[revision],'license':'project-generated',
                    'gold_sources':['constructed-world-oracle'],
                    'generation':{'generator':revision,'independent_world':group,
                        'authorship':'fresh constructed service facts, no held input or held query used',
                        'provider_calls':0,'teacher_trajectory':False},
                    'curriculum':{'version':'natlang.inline_curriculum/1','family':'fresh_service_recovery',
                        'family_version':1,'shape':f'{domain}-{shape}','variant':variant,'pair_group':None,
                        'split_group':group,'slice':'folder_failure','domain':'relational','mode':'single_call',
                        'inline':'optional','evidence':{'world':[f'fresh {domain} service pages'],'retrieved':[],'background':[]},
                        'assumptions':[],'decisive':[],'plausible_actions':[],
                        'minimum_sequence':['read service pages','ask the typed policy function for each record',
                            'reconcile all selected IDs and values','write and read back the summary'],
                        'reference':{'root':[['eval',{'code':reference}],['return_result',{'status':'success','value':expected}]],
                            'children':children}},
                    'semantics':{'root':'summarize.nl','files':{'types.ts':TYPES,
                        'summarize.nl':'---\nargs:\n  policy: "Policy"\nreturns: "Summary"\nkind: directory-reducer\n---\n'
                            f'Summarize the {domain} records supplied by {namespace}.\n'
                            '1. Read every source page using the service interface.\n'
                            '2. Ask select_record to apply policy to each complete record.\n'
                            '3. Include every selected ID in lexicographic order. Compute total from their values and count from their number.\n'
                            '4. Write the complete Summary as JSON to summary.json, read it back, and return it.\n',
                        'summarize/select_record.nl':'---\nargs:\n  record: "Row"\n  policy: "Policy"\nreturns: "boolean"\n---\n'
                            'Return true exactly when record.category equals policy.category, record.active is true, '
                            'and record.value is at least policy.minimum. Otherwise return false.\n'},
                        'inputs':{'policy':policy},'expected':expected,
                        'folder_files':{'summary.json':'{}\n'},
                        'expected_files':{'summary.json':json.dumps(expected,separators=(',',':'))+'\n'},
                        'services':{namespace:service},'failure_seed':{'code':seeds[variant]}}}


TERMINAL_REVISION = 'authored-terminal-policy-recovery-20261018/1'


def terminal_policy_worlds(shapes=1):
    """Fresh complete service facts, explicit exceptions, and terminal child answers."""
    # The existing constructor remains the only parent/service/recovery mechanism.
    # Changing its revision before construction gives independent IDs and inputs.
    # Pass the revision explicitly rather than mutating globals used by old worlds.
    domains = {name: DOMAINS[name] for name in ('inventory', 'builds', 'support')}
    for original in worlds(shapes, revision=TERMINAL_REVISION, domains=domains):
        row = original
        semantics = row['semantics']
        namespace, service = next(iter(semantics['services'].items()))
        prefix = 'const ROWS:Row[] = '
        encoded = service.split(prefix, 1)[1].split(';\n', 1)[0]
        records = json.loads(encoded)
        rng = random.Random(row['source_groups'][0] + ':exceptions')
        for record in records:
            record['blocked'] = rng.random() < .25
            record['waiver'] = rng.random() < .3
        # Ensure every individual world actually distinguishes each exception.
        policy = semantics['inputs']['policy']
        for i, (blocked, waiver, active, value) in enumerate((
                (True, True, True, policy['minimum'] + 1),
                (False, True, False, 0),
                (False, False, True, policy['minimum']),
                (False, False, True, policy['minimum'] - 1))):
            records[i].update(category=policy['category'], blocked=blocked,
                              waiver=waiver, active=active, value=value)
        predicate = ('record.category === policy.category && !record.blocked && '
                     '(record.waiver || (record.active && record.value >= policy.minimum))')
        selected = [record for record in records if
                    record['category'] == policy['category'] and not record['blocked']
                    and (record['waiver'] or (record['active'] and record['value'] >= policy['minimum']))]
        expected = {'ids': sorted(r['id'] for r in selected),
                    'total': sum(r['value'] for r in selected), 'count': len(selected)}
        types = TYPES.replace('active:boolean', 'active:boolean; blocked:boolean; waiver:boolean')
        semantics['files']['types.ts'] = types
        semantics['services'][namespace] = service.replace(TYPES, types, 1).replace(
            prefix + encoded, prefix + json.dumps(records, separators=(',', ':')), 1)
        semantics['files']['summarize/select_record.nl'] = (
            '---\nargs:\n  record: "Row"\n  policy: "Policy"\nreturns: "boolean"\n---\n'
            'Decide from the supplied complete record and policy. Return true exactly when '
            'the category matches and the record is not blocked, and either its waiver is true '
            'or it is active with value at least policy.minimum. A blocked record is always excluded. '
            'Return the resulting boolean directly.\n')
        semantics['expected'] = expected
        semantics['expected_files']['summary.json'] = json.dumps(expected, separators=(',', ':')) + '\n'
        if row['curriculum']['variant'] == 'typed-argument-recovery':
            semantics['failure_seed']['code'] = (
                'return await select_record({id:"seed",category:policy.category,'
                'value:"unavailable",active:true,blocked:false,waiver:false},policy);')
        reference = row['curriculum']['reference']
        reference['root'][-1][1]['value'] = expected
        reference['children'] = [{'match': ['select_record(', record['id']], 'calls': [
            ['eval', {'code': 'return ' + predicate + ';'}],
            ['return_result', {'status': 'success', 'value': record in selected}]]}
            for record in records]
        row['family'] = 'curriculum_fresh_terminal_policy_recovery'
        row['curriculum']['family'] = 'fresh_terminal_policy_recovery'
        row['curriculum']['assumptions'] = ['All policy facts are in the supplied record and policy.']
        yield row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protection',type=Path,required=True);p.add_argument('--shapes',type=int,default=16)
    p.add_argument('--profile', choices=('legacy', 'terminal-policy'), default='legacy')
    a=p.parse_args()
    if a.shapes<1:raise ValueError('shapes must be positive')
    if a.output.exists():raise ValueError('new teacher packet output must not exist')
    protection=current_protection(a.protection)
    root=Path(__file__).resolve().parents[2]
    holds,bindings=load_decision_holds(sorted((root/'training/decision-holds').glob('*.jsonl')))
    held_programs={identifier.split(':decision:',1)[0] for identifier in holds}
    rows=list((terminal_policy_worlds if a.profile == 'terminal-policy' else worlds)(a.shapes));groups={group for row in rows for group in row['source_groups']}
    if groups & (protection['held']|protection['quality']|protection['pending']):
        raise ValueError('fresh authored worlds intersect current protection')
    if {row['id'] for row in rows}&held_programs:raise ValueError('fresh program ID intersects decision holds')
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('duplicate authored world IDs')
    a.output.mkdir(parents=True)
    source=a.output/'cases.jsonl'
    source.write_text(''.join(json.dumps(row,sort_keys=(a.profile == 'terminal-policy'),ensure_ascii=False,separators=(',',':'))+'\n' for row in rows))
    write_json_durable(a.output/'selection-proof.json',{'schema':'natlang.fresh-authored-teacher-packet/1',
        'cases':len(rows),'cases_sha256':sha256_file_hex(source),'worlds':len(groups),'revision':TERMINAL_REVISION if a.profile == 'terminal-policy' else REVISION,
        'profile':a.profile,
        'domain_counts':dict(Counter(row['curriculum']['shape'].rsplit('-',1)[0] for row in rows)),
        'family_shape_counts':dict(Counter(row['curriculum']['variant'] for row in rows)),
        'protection':protection['proof'],'decision_hold_bindings':bindings,'held_and_pending_intersections':0,
        'generator_sha256':sha256_file_hex(__file__),'canonical_programs_sha256':canonical_json_sha256_hex(rows),
        'source_corpora':[],'training_admission':False,
        'qualification':'actual complete reference replay required before queue; actual teacher traces require independent admission'})


if __name__=='__main__':main()
