# Keep the exact sealed RL input on resume

The existing whole-episode collection policy and task selection remain unchanged.
A pending collection resumes its exact actor, task objects, order, seeds and bytes.

1. The existing durable selection snapshot stores the selected task objects.
2. If the collection input file already exists, authenticate its ordered objects
   against that snapshot using the shared canonical JSON identity. Retain its
   original bytes; the collection receipt independently verifies their exact SHA.
3. Only create the input file when absent, with the existing JSONL serialization
   and atomic durable write. Changed objects or order fail before collection.

This is crisp identity and durability plumbing. JSON dictionary key ordering in
an equivalent control snapshot must not silently change the bytes of an already
sealed collection. No task is completed, rewarded, dropped or recollected here.
