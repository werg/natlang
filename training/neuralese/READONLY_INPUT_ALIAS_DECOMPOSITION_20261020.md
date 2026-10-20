# Declared input aliases across existing namespaces

The existing training policy keeps all consumed data and exact cache bindings
available for continuation and publication. A source cache may be named beneath
an explicitly declared read-only root while its identical physical file resides
in the canonical production namespace. Both locations are already authorized.

1. Keep the declared lexical path and exact content/size binding.
2. Require that lexical external path to belong to a declared read-only root.
3. Require its resolved target to belong to the production namespace or a
   declared read-only root. A target outside their union remains an error.
4. Apply the same verifier when collecting dependencies and checking staged
   assets. Checkpoint dependencies retain their separate namespace rules.

This extends the existing crisp file verifier; it introduces no new training
policy, prompt, source admission, parser identity or model computation. The
current LFM publication failure is reproduced by a read-only row-index alias
pointing to its exact canonical production cache. Frozen runtimes require an
explicit successor before using the correction.
