Reference modeling code from `deepgrove/maple-preview` at `ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07` (MIT, see
LICENSE), used only by `test_maple_model.py` as the parity reference. `fa3.py` is replaced by an SDPA stand-in with
the same window rule (`window_size=(w, 0)`: keys at distance <= w).
`modeling_maple.py` is patched in one place: Transformers 5 removed `ROPE_INIT_FUNCTIONS["default"]`, so the rotary inverse frequencies are computed inline exactly as Transformers 4.57 did.
