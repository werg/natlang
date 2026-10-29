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


def resolve_object(api, receipt, role, digest, *, get):
    """Return an exact object or a confirmed absence, including tombstones."""
    identity=receipt['identity']
    if receipt.get('private') is not True or receipt.get('role')!=role or identity.get('sha256')!=digest:raise ValueError('provider deletion identity differs')
    values=list(api.run(receipt['run']).files(names=[receipt['name']]))
    if not values:return None
    if len(values)!=1 or values[0].name!=receipt['name']:raise ValueError('provider object identity differs')
    obj=values[0]
    if obj.size==0:
        response=get(obj.url,headers={'Range':'bytes=0-1'},timeout=15,stream=True)
        status=response.status_code;response.close()
        if status==404:return None
        raise ValueError('zero-byte listing not confirmed absent')
    if obj.size!=identity['bytes']:raise ValueError('provider object size differs')
    return obj
