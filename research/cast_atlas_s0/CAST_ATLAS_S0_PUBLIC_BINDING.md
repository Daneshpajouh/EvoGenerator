# CAST-ATLAS-S0 public execution binding

**Frozen:** 2026-07-17 (America/Vancouver)  
**Private preregistration path:** `research_campaigns/2026-07-17-round2/prereg/CAST_ATLAS_S0_PREREG.md`  
**Private Git blob SHA:** `a7d7d7dac61130abb588b7c2006e1fca4c22d676`

This public binding identifies the already-frozen private preregistration before
any supplementary table, workbook schema, event count, or independent-condition
count is inspected.

## Binding decision rules

`EXECUTABLE_ATLAS_ROUTE` requires:

1. compatible event-level tables from at least two independent studies;
2. explicit or deterministically reconstructable condition, reference, bait,
   partner-coordinate, event-class, support, control/validation, and replicate fields;
3. at least 59 independent treated edit conditions after technical replicate,
   primer-orientation, control, and repeated-modality collapse;
4. leave-one-study-out identity;
5. outcome-independent nomenclature mapping;
6. a measurable missed-event or negative denominator;
7. agreement between independently written metadata/table parsers.

Fastest kills are fewer than 59 conditions, fewer than two compatible studies,
or absence of a predictor-independent missed-event denominator.

The execution is a data-readiness gate only. It may not estimate clinical risk,
rank editor safety, or support treatment, embryo/germline, or reproductive claims.
