# Archived exporter source shadowing

The deployed `drain-archive.py` lived beside an old copied `vibestudio_campaign.py`.
Python searched that script directory before `PYTHONPATH`, so the archive copy
shadowed the selected pinned producer. Wave 0035's two-field readiness marker
exposed the old implementation. Its exact source SHA is
`4166e4e090f1709e487f2abf1ba9ec2a7d543d87cb481f755403de249fe8638f`.

Inventory: the direct `remote_archive` export command, the deployed drainer,
`deploy-semantic-recovered-pipeline.py` and `export-recovered-current.py` use this
module. Direct CPU recovery exporters run outside the archive directory and used
their explicitly selected pinned module. Host JS capture, native materialization,
ingestion and fill are separate entry points; they do not import the sibling
archive campaign module. Inference engines do not select these Python producer
entry points.

Direct exports now execute the selected repository's script with explicit
`NATLANG_REPO`. Active archive wrappers must prepend that pin's producer directory
and assert the imported module's resolved filename. The copied module is retained
as historical source outside the import directory. Original rows, captures and
registry snapshots are retained; the modern ingester independently reopens their
manifests and validates native evidence before extraction. Historical markers are
not evidence that the wrapper itself ran from the claimed Git revision. An
additive component-provenance audit records the copied module and selected native
admission code separately.
