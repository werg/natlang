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
    args = parser.parse_args()
    from natlang_neuralese.train.native_execution_transition import (
        prepare_candidate, verify_files, publish_diagnostic, verify_candidate_publication)
    def reference(path, sha):
        return {'path':str(Path(path).resolve(strict=True)), 'sha256':sha}
    if args.operation == 'prepare':
        result = prepare_candidate(reference(args.parent, args.parent_sha256), output=args.output)
    elif args.operation == 'verify':
        result = verify_files(reference(args.parent, args.parent_sha256), reference(args.child, args.child_sha256))
    elif args.operation == 'publish-diagnostic':
        result = publish_diagnostic(args.state, proof_ref=reference(args.proof, args.proof_sha256), out=args.output)
    else:
        result = verify_candidate_publication(reference(args.publication, args.publication_sha256))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
