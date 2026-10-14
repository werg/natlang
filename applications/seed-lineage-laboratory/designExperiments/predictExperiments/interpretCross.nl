---
description: Resolve the actual parents and material compartment of one experiment.
args:
  experiment: ExperimentPlan
  pedigree: PedigreeReading
  assay: Assay
returns: CrossReading
---
1. Bind the actual experiment, returned accession inventory and assay. Declare CrossMeaning with
   oriented parent/subject fields, interpretable, reason and quote support. Directly await a scoped literal.
2. The literal identifies who supplies the ovule, pollen and any assayed tissue. It distinguishes
   reciprocal crosses and interprets whether the measurement concerns offspring or the original subject.
3. For a probe/image, retain its actual subject and absent cross parents. For a cross, require the
   original parent identities and source generation convention; missing orientation returns null.
4. Consume the returned meaning and bind original source witnesses. Check all decided accession IDs
   against the supplied pedigree and all nonempty quotes against the experiment/assay witnesses.
5. Return CrossReading. Its motherId governs later cytoplasm assignment; its compartment determines
   whether an offspring genotype can even influence the reported measurement.
