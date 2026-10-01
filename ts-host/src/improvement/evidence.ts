/** A read-only, pageable view; selecting context never discards the underlying evidence. */
export class EvidenceView<T> {
  private readonly items: readonly T[];
  constructor(items: readonly T[]) {this.items=structuredClone(items);Object.defineProperty(this,'items',{enumerable:false});}
  get length():number{return this.items.length;}
  page(start=0,limit=20):T[]{
    if(!Number.isSafeInteger(start)||start<0||!Number.isSafeInteger(limit)||limit<1||limit>200)throw new RangeError('evidence page requires start >= 0 and limit 1..200');
    return structuredClone(this.items.slice(start,start+limit));
  }
}
