---
description: Incorporate one source dependency and preserve its actual documentary scope.
args:
  state: FamilyFrontier
returns: FamilyFrontier
---
1. Return state if pendingSourceIds is empty. Bind sourceId to its first original ID and role to the
   unique actual WitnessRole for it. Keep pending tail and existing families immutable until this step ends.
2. If role is missing or conflicting, retain an unresolved finding naming sourceId and the missing
   history. Consume only this pending ID, preserving existing family memberships for later audit.
3. Bind dependency to the actual role.copyOf and evidenceScope to role.scope. Read whether this history
   establishes copying of the notation/report being compared or only another documentary component.
4. For dependency=null with independent=true, find any provisional family already containing sourceId.
   Complete that existing membership rather than create a duplicate; otherwise start one independent
   family. Preserve role witnesses as its actual documentary basis.
5. For an established copying dependency, require the exemplar ID to exist in actual roles. Trace its
   source-established predecessor links over the finite original role collection, retaining visited IDs.
   A repeated visited source is a copying-history cycle; stop that trace and record the exact cycle.
6. Interpret scope along this actual chain. When every link covers the present notation/report scope,
   the copy and exemplar are dependent evidence. If a link covers only another component, retain the
   scope conflict as unresolved instead of making a global independence claim from partial history.
7. Locate all existing families containing the source or exemplar. Merge only the scope-compatible
   ones and union their original source IDs and exact witnesses. Add the current source/exemplar IDs;
   an exemplar still pending is provisional and its own later step must interpret its outgoing history.
8. If independent is null or a decisive copying link is unknown, retain the actual documentary question
   with witnesses. A separate provisional row can preserve coverage, but its scope/reason must expose
   uncertainty so comparison cannot count it as an established independent vote.
9. Check membership changes against only the current original source and established dependency chain.
   Preserve unrelated families; generated family IDs and family size have no authority meaning.
10. Remove exactly the selected pending ID and return actual families plus findings. Original ID/quote
    union is mechanical bookkeeping after the NL scope/dependency judgment; it cannot choose a reading.
