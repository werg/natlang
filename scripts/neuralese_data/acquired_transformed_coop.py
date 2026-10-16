"""The exact released two-actor transcript, never invented four-actor contexts."""
import csv
import json
from pathlib import Path

from .common import Reject
from .records import group_key


def qwen_comm4_acquired(limit=None):
    from . import agents
    from .acquired_agents import acquired_rows,_reported_session,_bound_acquired_path
    previous=None;index={};index_binding=None;known={}
    for receipt,rid,row in acquired_rows('D15',limit):
        try:
            if receipt is not previous:
                previous=receipt;known={e['path']:e for e in receipt['files']}
                index_binding=known.get('index.csv')
                if index_binding is None:raise Reject('transformed source lacks original index provenance')
                path=_bound_acquired_path(agents.EXPANSION_ROOT/'D15',index_binding)
                with path.open(newline='',encoding='utf-8') as stream:
                    values=list(csv.DictReader(stream))
                index={}
                for item in values:
                    if item['log_dir'] in index:raise Reject('ambiguous original transformed transcript index')
                    index[item['log_dir']]=item
            path=rid.rsplit(':',1)[0]
            if not path.endswith('/readable_merged.json'):
                raise Reject('transformation audit metadata; exact released transcript is the consumer')
            if not isinstance(row,dict) or not isinstance(row.get('entries'),list):
                raise Reject('transformed source lacks its complete ordered entry inventory')
            original=index.get(path)
            if original is None:raise Reject('transformed transcript lacks exact original index row')
            parent=Path(path).parent;binding=known.get(str(parent/'transform_metadata.json'))
            if binding is None:raise Reject('transformed transcript lacks bound original transformation audit')
            audit=json.loads(_bound_acquired_path(agents.EXPANSION_ROOT/'D15',binding).read_text())
            parts=parent.parts
            if (len(parts)<3 or str(original['repo'])!=parts[-3] or str(original['task_id'])!=parts[-2]
                    or str(original['features'])!=parts[-1].replace('f','')):
                raise Reject('transformed index belongs to another original benchmark task')
            entries=row['entries'];actors=[]
            if original.get('n_turns') and int(original['n_turns'])!=len(entries):
                raise Reject('transformed entry inventory differs from original index')
            if (audit.get('stats') or {}).get('total_entries_out') not in (None,len(entries)):
                raise Reject('transformed entry inventory differs from original transformation audit')
            for entry in entries:
                if (not isinstance(entry,dict) or not isinstance(entry.get('agent'),str)
                        or entry.get('actor')!='CODER-'+entry['agent'] or entry.get('type') not in ('action','thinking','observation')
                        or not isinstance(entry.get('content'),str)):
                    raise Reject('unsupported original transformed entry; no context may be dropped')
                if entry['agent'] not in actors:actors.append(entry['agent'])
            meta=json.loads(original.get('metadata') or '{}')
            for actor in actors:
                messages=[{'role':'user','content':f'Continue the released transformed coding transcript for {actor} on {original["repo"]}:{original["task_id"]} ({original["features"]}).'}]
                mapping=[]
                for ordinal,entry in enumerate(entries):
                    kind=entry['type'];body=entry['content'];own=entry['agent']==actor
                    # The release labels exit-status records as action; they
                    # remain observed context rather than generated tool output.
                    observed=(not own or kind=='observation' or kind=='action' and body.startswith('(result: exit '))
                    if observed:
                        message={'role':'tool','content':json.dumps(entry,ensure_ascii=False,separators=(',',':'))}
                    elif kind=='thinking':message={'role':'assistant','content':'','reasoning':body}
                    else:message={'role':'assistant','content':body}
                    mapping.append({'entry_index':ordinal,'message_index':len(messages),'original':entry})
                    messages.append(message)
                child=rid+':agent:'+actor
                t=_reported_session('D15',receipt,child,{},messages,task_id=f'{original["repo"]}:{original["task_id"]}',
                    session_id=child,teacher=(row.get('run_info') or {}).get('model') or original.get('model'))
                t.messages=messages
                status=(meta.get('agent_statuses') or {}).get(actor)
                if status and status not in ('Submitted','Completed','success'):t.outcome['label']='failed'
                if original.get('passed','').strip().lower() in ('false','0'):t.outcome['label']='failed'
                t.split_groups.append(group_key('cooper-task',f'{original["repo"]}:{original["task_id"]}'))
                if f'{original["repo"]}:{original["task_id"]}' in agents._bench_ids():t.split='test'
                t.lineage['notes']['original_transformed_cooperation']={'index_binding':index_binding,'index_row':original,
                    'transform_binding':binding,'transform_audit':audit,'run_info':row.get('run_info'),
                    'actor':actor,'entries':mapping,'whole_released_view':True,'original_four_agent_contexts_available':False,
                    'source_test_result_available':bool(original.get('passed')),'executed_locally':False}
                t.outcome['details'].update(reported_status=status,reported_passed=original.get('passed'),executed_locally=False)
                yield child,t
        except (Reject,ValueError,TypeError,KeyError) as error:yield rid,Reject(str(error))
