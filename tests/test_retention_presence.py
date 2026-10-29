import pytest
from scripts.coreweave.retention_presence import current_presence


def test_grouped_fresh_provider_listing_fails_on_missing_duplicate_or_changed():
    class Obj:
        def __init__(self,name,size):self.name=name;self.size=size
    class Api:
        def __init__(self):self.calls=[];self.values=[Obj('a',2),Obj('b',3)]
        def run(self,r):self.calls.append(r);return self
        def files(self,*,names):assert names==['a','b'];return self.values
    api=Api();receipts=[{'private':True,'run':'r','name':n,'identity':{'bytes':b}} for n,b in [('a',2),('b',3)]]
    assert current_presence(api,receipts) and api.calls==['r']
    for values in ([Obj('a',2)],[Obj('a',2),Obj('b',4)],[Obj('a',2),Obj('a',2)]):
        api.values=values
        with pytest.raises(ValueError):current_presence(api,receipts)


def test_tombstone_requires_exact_404_confirmation():
    from scripts.coreweave.retention_presence import resolve_object
    class Obj:name='n';size=0;url='https://example.invalid/object'
    class Api:
        def run(self,r):return self
        def files(self,*,names):return [Obj()]
    class Reply:
        def __init__(self,status):self.status_code=status
        def close(self):pass
    receipt={'private':True,'role':'x','run':'r','name':'n','identity':{'sha256':'a'*64,'bytes':12}}
    assert resolve_object(Api(),receipt,'x','a'*64,get=lambda *a,**k:Reply(404)) is None
    with pytest.raises(ValueError,match='not confirmed'):resolve_object(Api(),receipt,'x','a'*64,get=lambda *a,**k:Reply(200))
    with pytest.raises(ValueError,match='identity'):resolve_object(Api(),receipt,'x','b'*64,get=lambda *a,**k:Reply(404))
