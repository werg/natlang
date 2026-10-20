# Historical classifier identity across runtime locations

Policy: retain admitted source history and its actual producer provenance when
moving to a newer runtime. A directory change alone must not invalidate earned
training. A different classifier implementation is still a different producer.

1. Bind the currently trusted classifier's exact SHA-256 and byte count.
2. Require those exact code bytes in every historical proof's producer binding.
3. Independently authenticate that producer at its original recorded path. Keep
   that original binding in the proof; never substitute a current runtime path.
4. Preserve all existing source, held, index, seven-policy-file and metadata-only
   work checks. No graph payload is reparsed and no visits are added.

This extends the existing recipe proof verifier, an exact crisp mechanism. It
changes neither classification policy nor the classifier producer itself.
