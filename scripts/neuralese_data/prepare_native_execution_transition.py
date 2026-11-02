#!/usr/bin/env python3
"""Prepare, verify, or register a held zero-SGD native execution candidate."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    prepare = commands.add_parser('prepare')
    for flag in ('parent', 'parent-sha256', 'output'):
        prepare.add_argument('--' + flag, required=True)
    verify = commands.add_parser('verify')
    for flag in ('parent', 'parent-sha256', 'child', 'child-sha256'):
        verify.add_argument('--' + flag, required=True)
    publish = commands.add_parser('publish-diagnostic')
    for flag in ('state', 'proof', 'proof-sha256', 'output'):
        publish.add_argument('--' + flag, required=True)
    check = commands.add_parser('verify-candidate')
    for flag in ('publication', 'publication-sha256'):
        check.add_argument('--' + flag, required=True)
    qualification = commands.add_parser('qualify')
    for flag in ('publication', 'publication-sha256', 'evidence', 'evidence-sha256', 'protection-root', 'output'):
        qualification.add_argument('--' + flag, required=True)
    verify_qualification = commands.add_parser('verify-qualification')
    for flag in ('qualification', 'qualification-sha256'):
        verify_qualification.add_argument('--' + flag, required=True)
    select = commands.add_parser('publish-selected')
    for flag in ('state', 'qualification', 'qualification-sha256', 'decision-reason', 'output'):
        select.add_argument('--' + flag, required=True)
    verify_selected = commands.add_parser('verify-selected')
    for flag in ('publication', 'publication-sha256'):
        verify_selected.add_argument('--' + flag, required=True)
    seal = commands.add_parser('seal-collection')
    for flag in ('out', 'request', 'request-sha256', 'output'):
        seal.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    from natlang_neuralese.train.native_execution_transition import (
        prepare_candidate, verify_files, publish_diagnostic, verify_candidate_publication,
        qualify_candidate, verify_qualification as verify_qualification_receipt,
        publish_selected, verify_selected_publication, seal_fresh_collection)
    def reference(path, sha):
        return {'path':str(Path(path).resolve(strict=True)), 'sha256':sha}
    if args.operation == 'prepare':
        result = prepare_candidate(reference(args.parent, args.parent_sha256), output=args.output)
    elif args.operation == 'verify':
        result = verify_files(reference(args.parent, args.parent_sha256), reference(args.child, args.child_sha256))
    elif args.operation == 'publish-diagnostic':
        result = publish_diagnostic(args.state, proof_ref=reference(args.proof, args.proof_sha256), out=args.output)
    elif args.operation == 'verify-candidate':
        result = verify_candidate_publication(reference(args.publication, args.publication_sha256))
    elif args.operation == 'qualify':
        from natlang_neuralese.common.jsonio import write_json_durable
        out = Path(args.output).resolve()
        if out.exists():
            raise ValueError('execution qualification receipt already exists')
        result = qualify_candidate(reference(args.publication, args.publication_sha256),
            reference(args.evidence, args.evidence_sha256), protection_root=Path(args.protection_root).resolve(strict=True))
        write_json_durable(out, result)
    elif args.operation == 'verify-qualification':
        result = verify_qualification_receipt(reference(args.qualification, args.qualification_sha256))
    elif args.operation == 'publish-selected':
        result = publish_selected(args.state, qualification_ref=reference(args.qualification, args.qualification_sha256),
                                  out=args.output, decision_reason=args.decision_reason)
    elif args.operation == 'verify-selected':
        result = verify_selected_publication(reference(args.publication, args.publication_sha256))
    else:
        result = seal_fresh_collection(out=args.out, request_ref=reference(args.request, args.request_sha256),
                                       proof_out=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
