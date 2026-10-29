"""One fresh provider listing per run for exact retention object identities."""
from collections import defaultdict


def current_presence(api, receipts):
    """Fail closed on duplicate identities, absent entries or mismatched sizes."""
    groups=defaultdict(list)
    for receipt in receipts:
        if receipt.get('private') is not True:raise ValueError('object receipt is not private')
        identity=receipt.get('identity') or {}; size=identity.get('bytes')
        if type(size) is not int or size<=0 or not receipt.get('run') or not receipt.get('name'):
            raise ValueError('invalid object receipt')
        groups[receipt['run']].append(receipt)
    for run,items in groups.items():
        names=[x['name'] for x in items]
        if len(set(names))!=len(names):raise ValueError('duplicate object identity')
        objects=list(api.run(run).files(names=names));found={}
        for obj in objects:
            if obj.name not in names or obj.name in found:raise ValueError('unexpected provider listing')
            found[obj.name]=obj
        for item in items:
            obj=found.get(item['name'])
            if obj is None or obj.size!=item['identity']['bytes']:
                raise ValueError('required private object missing or changed: '+item['name'])
    return True
