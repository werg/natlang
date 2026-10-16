"""Prepare an immutable derivative proposal with fit-only training entrypoints."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_pending_sources import load_pending_sources, bound as pending_bound
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions
from scripts.neuralese_data.prepare_acquired_session_admission import bound


def propose(full_proposal, partition_receipt, declarations, *, cohort_root, identifier, output, protection_continuation=None):
    root=Path(cohort_root).resolve(strict=True)
    original=json.loads(Path(full_proposal).read_text())
    receipt_path=Path(partition_receipt).resolve(strict=True);directory=receipt_path.parent
    receipt=json.loads(receipt_path.read_text())
    if (identifier==original['id'] or receipt.get('schema')!='natlang.acquired-native-capacity-partition/1'
            or receipt.get('source_eligibility_unchanged') is not True
            or receipt.get('exact_original_row_bytes') is not True or receipt.get('execution_qualified') is not False
            or receipt.get('binding',{}).get('audit',{}).get('sha256')!=original['build']['audit_sha256']
            or set(declarations)!=set(receipt['counts'])):
        raise ValueError('capacity derivative requires the complete original source eligibility and a new immutable corpus ID')
    bound(root/original['build']['audit'],{'sha256':original['build']['audit_sha256']})
    for item in receipt['files']:bound(directory/item['path'],item)
    for item in receipt['source_bindings']:bound(item['path'],item)
    include=set(original['include'])
    for relative in include:
        if not (root/relative).is_file():raise ValueError('capacity derivative lost original source evidence')
    include.add(str(receipt_path.relative_to(root)))
    include.update(str((directory/item['path']).relative_to(root)) for item in receipt['files'])
    pending_ids={model:set() for model in declarations}
    for row in sessions(directory/receipt['coverage']):
        if row['split']=='train' and row['partition']=='remaining-long':
            if row['id'] in pending_ids[row['model']]:raise ValueError('capacity coverage repeats a pending row')
            pending_ids[row['model']].add(row['id'])
    entrypoints={};documents={}
    for model,path in declarations.items():
        path=Path(path).resolve(strict=True);raw=json.loads(path.read_text())
        loaded=load_pending_sources(path,tokenizer_fingerprint=raw['tokenizer_fingerprint'])
        expected=directory/receipt['manifests'][model+':train:remaining-long']
        if (set(loaded['rows'])!=pending_ids[model] or any(row['required_visits']!=1 for row in loaded['rows'].values())
                or len(loaded['sources'])!=1
                or pending_bound(loaded['sources'][0]['pending_native_graphs'],path.parent)!=expected.resolve()
                or str(pending_bound(loaded['sources'][0]['full_cohort'],path.parent).relative_to(root)) not in include):
            raise ValueError('capacity derivative pending declaration does not cover the complete unchanged long-row set')
        include.add(str(path.relative_to(root)))
        entrypoints[model]={'native_graphs':str((directory/receipt['manifests'][model+':train:fit']).relative_to(root)),
            'pending_native_sources':str(path.relative_to(root)),
            'execution_qualified':False,'requires_actual_capacity_certificate':True}
        fit=receipt['counts'][model]['train:fit'];long=receipt['counts'][model]['train:remaining-long']
        documents[model]={'source_eligible_graphs':fit['rows']+long['rows'],
            'fit_train_graphs':fit['rows'],'fit_train_tokens':fit['tokens'],
            'pending_long_graphs':long['rows'],'pending_long_tokens':long['tokens'],'earned_graph_visits':0}
    proposal=copy.deepcopy(original);proposal.update(id=identifier,include=sorted(include),documents=documents,
        source_class='capacity-partition-of-admitted-complete-acquired-native-sessions',
        training_entrypoints=entrypoints,
        training_admission_reason='Unchanged source-curated rows; ONLY explicit fit entrypoints may train after actual capacity qualification. Full cohort is provenance, not an additional training source; long rows remain mandatory zero-credit obligations.')
    proposal['build']['capacity_partition']={'path':str(receipt_path.relative_to(root)),
        'sha256':sha256_file_hex(receipt_path),'source_corpus_id':original['id'],
        'full_cohort_is_additional_training_source':False,'execution_qualified':False}
    if protection_continuation is not None:
        from scripts.neuralese_data.continue_acquired_protection import verify_certificate
        certificate_path = Path(protection_continuation).resolve(strict=True)
        certificate = verify_certificate(certificate_path, audit_sha256=original['build']['audit_sha256'])
        for ref in certificate['dependencies']:
            include.add(str((certificate_path.parent / ref['path']).resolve(strict=True).relative_to(root)))
        include.add(str(certificate_path.relative_to(root)))
        proposal['include'] = sorted(include)
        proposal['build']['protection_continuation'] = {
            'path': str(certificate_path.relative_to(root)), 'sha256': sha256_file_hex(certificate_path),
            'new_protection': {**certificate['new_protection'], 'path': str(
                (certificate_path.parent / certificate['new_protection']['path']).resolve(strict=True).relative_to(root))},
            'eligibility_only': True, 'numeric_partition_unchanged': True}
        proposal['training_admission_reason'] += ' Eligibility continued explicitly under the bound stricter connected-source protection; original numeric audit and partition remain unchanged.'
    write_json_durable(output,proposal)
    return proposal


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('full-proposal','partition-receipt','cohort-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--id',required=True);parser.add_argument('--declarations',required=True,help='JSON model-to-path map')
    parser.add_argument('--protection-continuation',type=Path)
    args=parser.parse_args(argv)
    propose(args.full_proposal,args.partition_receipt,json.loads(args.declarations),cohort_root=args.cohort_root,
        identifier=args.id,output=args.output,protection_continuation=args.protection_continuation)
    return 0


if __name__=='__main__':raise SystemExit(main())
