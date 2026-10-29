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
