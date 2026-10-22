# Exclusive feeder custody per verified teacher actor

The owner selected Qwen3.6 and authorized continuing useful generation while
existing Flint cases finish. Keep each teacher's generation root, source epoch,
execution identity, queues and admission namespace distinct. GPU serving
allocations remain explicit; this change does not allocate or resize a server.

1. Authenticate the selected teacher's immutable execution-identity binding and
   its actual live process identities before reserving a feeder launch.
2. Under one short process-launch lock, inspect existing actual feeder processes.
   Read only their declared teacher environment fields and authenticate their
   source descriptor, execution identity and live actor processes through the
   existing verifier. Recheck feeder identity after reading its environment.
3. A feeder using any of the selected actor's processes conflicts even if it
   declares a different endpoint or model alias. An unknown or changed identity
   remains a conflict. Only fully authenticated disjoint actors may overlap.
4. Launch and publish the existing prepared feeder under the same short lock,
   closing the scan/spawn race. Release the lock after owner publication; useful
   generation by distinct actors continues concurrently. Existing batch custody,
   process-token recovery, source selection, drain, transport and admission
   checks remain unchanged.

These steps are exact process custody and authentication mechanisms. The
existing natural-language operations policy still chooses tasks and teachers.
There is no model-facing instruction, sampling, trajectory or reward change.
Tests exercise actual subprocess identities/environment, changed source bytes,
endpoint aliases of the same actor, unknown owners and normal legacy recovery.
