# Capacity index witness verification

The typed operation verifies every declared request independently, then verifies
its warmed source inventories once. Exact canonical witnesses, including cache
bindings, complete keys and entry metadata digests, define reusable work.

1. Retain each request, source, case and reference declaration unchanged.
2. Form an ordered union of complete index witnesses across those requests.
3. Run the existing index verifier once on that union. Distinct conflicting
   witnesses remain distinct and fail the normal verifier.
4. Continue fresh numerical qualification for every declared case.

Union construction and the existing verifier are crisp exact identity mechanisms;
this introduces no scheduling or qualification policy. It changes no source rows,
parser keys, capacity requirements, model weights, optimizer, RNG or exposure.
