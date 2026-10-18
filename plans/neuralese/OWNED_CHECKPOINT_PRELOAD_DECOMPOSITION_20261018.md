# Owned checkpoint preload

The supervisor may overlap the necessary parent decode with source preparation.
It must keep one owner, validate the final training declaration before allocating
the model, and retain every normal restoration and qualification check.

1. The existing checkpoint policy loads the exact parent once on CPU, using its
   existing mmap and shared-participant behavior. Bind path, file stat and SHA.
2. Retain that state in a one-use authority. Reject a different path, participant,
   changed source or second consumption. Release its state reference on consumption.
3. The ordinary native checkpoint loader optionally consumes this authority in
   place of its initial file decode. All subsequent model/config/sparse-row and
   full-state restoration checks remain the same. Actor-view and default loading
   keep their existing paths.

The supervisor's timing decision remains natural language. The one-use source
authority and exact loader branch are crisp durability and restoration plumbing;
they introduce no model arithmetic, training credit or new checkpoint format.
