# Frozen orbital population input

`active_gp_20261102.json.gz` is a gzip-compressed CelesTrak active GP snapshot
retrieved on 2026-11-02 from:

https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json

The uncompressed JSON has 16,847 records and SHA-256:

`87ee0c4a0a94681f299cdd6fa9c84ae429b78359f4b794ba358e5df11b766878`

Source/schema documentation:
https://celestrak.org/NORAD/documentation/gp-data-formats.php

```sh
gzip -dc experiments/rust_online/data/active_gp_20261102.json.gz > /tmp/frozen_active_gp.json
```

This is population input for synthetic experiments, not measured sensor data.
Scenario manifests record which source rows were resampled, their synthetic
perturbations, the SGP4 propagation contract, and assumed observation noise.
