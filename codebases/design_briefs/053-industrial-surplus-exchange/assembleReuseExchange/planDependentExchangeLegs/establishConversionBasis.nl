---
args: {"profile": "MaterialProfile", "need": "FunctionalNeed", "events": "Evidence[]"}
returns: "string[]"
---
Compare offered and required dimensions. Weight and volume require a measured density with applicable material, moisture/condition and unit basis; never use a generic conversion from memory. Equipment counts cannot be converted to mass without actual unit data. Return the source-bound rational conversion and allowed uncertainty or the precise missing measurement. Keep unconverted quantities in their original units until the basis is authorized.
