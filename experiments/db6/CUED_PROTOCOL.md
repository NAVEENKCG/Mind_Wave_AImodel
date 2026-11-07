# DRAFT - awaiting user approval

## Candidate freeze protocol

This is a candidate process document for review. It is not adopted, and it does
not select an operating confidence threshold.

### Candidate methods and scope

- Candidate method: **cued-refit**.
- **cued-stats** remains a control only. It uses labels to select calibration
  windows for frozen statistics and does not retrain the classifier.
- `raw` is the frozen D1T1 classifier baseline.
- This is a synthetic calibration/replay diagnostic assembled from complete
  labeled repetition blocks in the blocked DB6 recording. It is not
  chronological live deployment.
- The candidate calibration is supervised: the D1T1 training data (repetitions
  1-8) is augmented with the selected target-session calibration windows.
  The resulting LDA is frozen before held-out replay.
- LDA uses its default empirical class priors (`priors=None`) and no class
  weighting. This preserves the existing classifier definition; no weighting
  or prior was tuned on evaluation data.

### Split and replay

For each session, actual labels are inspected first. The calibration pool is
the first four common repetitions per grasp (1-4); the common held-out set is
repetitions 5-12 per grasp. Grasp identity is part of the key, so repeated
repetition numbers across grasps are not merged. Calibration and evaluation
blocks are disjoint in both `(session, grasp, repetition)` identity and raw
sample range.

For each synthetic block, causal filtering and replay state start fresh.
Windows are built only inside one original contiguous block; no analysis
window crosses a synthetic join. Calibration windows are never scored.

The exploratory options remain:

- `k=2`: 65.0% accuracy, 5.48 false events/min, approximately 2 minutes of
  recorded-block calibration proxy.
- `k=4`: 67.3% accuracy, 4.48 false events/min, approximately 4 minutes of
  recorded-block calibration proxy.

These are S1 development results, not independent validation. Both `k=2` and
`k=4` remain on the table pending the user's choice.

### Metrics and fixed sweep

The fixed confidence sweep is `tau = 0.0, 0.6, 0.8`; all values are reported.
No operating value is selected here.

- Accuracy and balanced accuracy are computed over valid, held-out scored
  windows.
- `miss` is **missed active windows**: the fraction of scored windows with a
  non-rest label whose final prediction is REST. It is not a grasp-event
  miss rate.
- `rest_false` is the fraction of scored REST windows predicted as a grasp.
- `events_min` counts starts of runs of majority-smoothed grasp output during
  valid REST windows in each independent replay block. Its denominator is the
  duration of those valid REST windows only:
  `count * 0.1 seconds / 60`. The current cued scorer does not apply the
  replay harness's release guard, so release-lag windows remain in this
  denominator and can contribute to the event count.
- Rejection is identical for every method: confidence below `tau` becomes
  REST. A rejected active window therefore contributes to `missed active
  windows` identically for every method.
- `calib_seconds` is the summed observed duration of selected recorded blocks
  per grasp. It is a lower-bound calibration-time proxy, excluding setup,
  cueing, and instruction overhead.
- `elapsed_original_seconds` is the span between the earliest selected block
  start and latest selected block end in the original blocked recording. It is
  not the calibration-time proxy.

### Threshold and validation rules

No operating threshold may be selected from target-session evaluation data.
Any future threshold decision requires a prespecified objective computed on
validation data only, before looking at held-out evaluation results.

If and when the user approves this freeze, any further subjects must be run
once with this frozen protocol, without retuning. No additional subjects are
run or simulated as part of this draft.

### Reporting

Report every target session, every `k`, every method, and all three fixed
threshold values. Read false activations next to missed active windows;
rejecting more predictions can reduce false activations while increasing
misses. Keep the S1 results labeled as development evidence.
