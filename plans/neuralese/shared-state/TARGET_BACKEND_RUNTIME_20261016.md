# Target backend qualification runtime

This isolated runtime starts from qualified operational runtime 69c888c682bc. It adds the published FP8 scope/actual-execution counters (631a39b2a), explicit persisted BF16 expert input adjoints (464621f04), chronological sparse attention traversal from b5cff288d, and the recipe qualification guard bcf9fb1d1. Prefix sharing stays absent/off. All seven admission-policy files remain byte-identical, so exact prepared source indexes remain usable.

The supervisor chooses a useful saved-parent handoff. The shared strict restored-model verifier performs unchanged numerical gates inside the requested backend scope, requires actual executed projection counts, and restores RNG/backend statistics. No optimizer update is used as a qualification shortcut. An unsuccessful candidate leaves the qualified operational runtime available for exact-state continuation.

Local focused tests and quick checks establish composition only. Production activation requires the actual trained Mellum parent in NGC2.14/cu134 on SM120, unchanged tolerances, positive actual execution counts and then useful updates. Local SM121 evidence does not substitute for that target check.

Admission file SHA-256 inventory:

```json
{
  "data/native_sources.py": "15b43272e79acf2b73f3793416a87a1b844850f8df22c2c2fe27870c719abe97",
  "data/native_graph_schema.py": "8ad9a94a5429080b3e6e407982517c3bb8cd2f83d2b2bf436e9ce72662e575c3",
  "model/native_graph_plan.py": "80fa45cadbb455effcbf55260d7a2c31fae0c1cedb5b5dc52ac497e0527a01cc",
  "data/native_pair_schema.py": "839df4000876a6c0f9ac1283f14fbae07cd66188e170d618b4e6008b8715f91d",
  "common/row_index.py": "e592aa380417424c3018f6d1672a2b961e45c1571a7e49bdec9bc4be11123544",
  "common/jsonio.py": "0716177d8690b53f3ea39f6e6dd3c1ebe5ff7407f02a7d095e9e59d4a1631114",
  "common/hashing.py": "736421674d6c1d9505f14f996e151ef320962bcef80bb0cf741dd9f686dce100"
}
```
