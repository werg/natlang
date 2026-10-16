# Ordered teacher system messages

The pinned NVIDIA Qwen chat template rejects later system messages in real
natlang recovery conversations. This is a transport contract failure, not
evidence of model inability. Preserve the actual message roles, order and text.

1. Derive a separately hashed template from the pinned source. Replace only its
   exact later-system rejection with the same system role delimiters and its
   existing system-content renderer. Fail if the expected source guard differs.
2. Keep all originally accepted conversations byte-identical, including the
   original first-system/tool/reasoning rendering. Later system messages remain
   at their original positions; never move or combine recovery instructions.
3. Test the pinned source and derived renderer on actual and varied conversation
   shapes. Bind source/derived hashes and engine arguments into actor provenance.
4. Qualify the changed rendering in the standing live teacher window before
   adopting it for generation. Continue useful generation on the qualified
   serving path while this is prepared. Static checks are not task-lift evidence.

This is exact serialization plumbing. Natlang's recovery instructions and
decisions remain in the existing natural-language functions and runtime errors.
