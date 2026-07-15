# Output validity after scatter-parameter audit

## Decision

The audited outputs below remain valid for their stated purpose. The active cache metadata and active configs carry the production values. No output is marked `INVALID_REGENERATE_REQUIRED` from this audit.

The cache metadata check was performed on the available v33e `F3inv_Vsel` caches. They report `xi=3.444`, `waves_vec_1=[0,1]`, `waves_vec_2=[0]`, `scatter1_00=4.04`, `scatter1_10=-43.2`, and `scatter2_00=4.12`. Available cache windows report `[0.2631, 0.36]`.

| output | validity | evidence | limitation |
|---|---|---|---|
| L20/000_A1m accepted windows | VALID | derived from corrected raw-grid outputs and user labels; corresponding cache metadata are production-correct | accepted-window provenance remains user-label based |
| L24/000_A1m accepted windows | VALID | reused completed corrected CPU raw scan; corresponding cache metadata are production-correct | accepted-window provenance remains user-label based |
| L20/100_A2 accepted windows | VALID | corrected-reader CPU/GPU reference path; GPU bridge and classifier identities passed; cache metadata are production-correct | no new scan was run here |
| L20/110_A2 accepted windows | VALID | corrected raw determinant grid/predictions and passed labels; cache metadata are production-correct | classifier acceptance remains limited to the scored labels |
| L24/100_A2 bracket 36 review | VALID | review uses the existing corrected raw/projected-basis determinant grid; cache metadata are production-correct | bracket 36 remains user-labeled false and classifier score remains FP=1; no label was changed |
| L20/111_A2 validation files | VALID / UNLABELED | corrected raw grid and predictions use the production-correct cache metadata | no accepted classifier claim until user labels are supplied |
| projected-basis prototype | VALIDATED PROTOTYPE | report records corrected v33h reader, production K3df point, direct-vs-basis raw determinant/sign agreement | prototype is not a replacement for raw determinant output |
| GPU determinant bridge validation | VALID | explicit CLI uses corrected reader, fixed production K3df values, and passed CPU reference identity checks; cache metadata are production-correct | bridge validation is for L20/100_A2 only |

## Why no regeneration is required

The determinant scanners and GPU bridge take the corrected cache directly and do not load `config_v31zw_*`. Classifier sweeps are CSV-only. The current fitter input is separately audited and contains the production values. Therefore the archival comparison values cannot be traced into these outputs.

No cachegen or determinant scan was run as part of this audit.
