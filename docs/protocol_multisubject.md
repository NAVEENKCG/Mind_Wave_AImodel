# NinaPro DB6 multi-subject protocol

This protocol is frozen before any held-out-subject run. Subject 1 is development
data (`data/db6_subject`); subjects 2, 3, and 4 are held out
(`data/db6_subjects/s2`, `s3`, and `s4`). Subjects are processed independently.
No subject 2–4 data is used to tune a threshold, feature, or classifier.

## Fixed signal and classifier

- Causal fourth-order Butterworth bandpass: 20–250 Hz.
- Raw sampling rate 2000 Hz, decimated to 1000 Hz.
- 200 ms windows with a 100 ms step.
- Features: mean absolute value, waveform length, zero crossings, and slope sign
  changes.
- Zero-crossing and slope-sign-change threshold: 0.02 times the training-window
  standard deviation.
- LDA classifier, unchanged from `experiments/db6/db6_replay.py` and
  `experiments/db6/db6_calib.py`.

## Subject and session split

For each subject, the first `(day, session)` file must be `D1 T1` and is that
subject's training session. Training uses repetitions 1–8 of every grasp in
that file. Test repetitions are 9–12.

Calibration uses repetitions 1 through `k` of each grasp, with
`k ∈ {0, 1, 2, 4}`. Calibration windows are excluded from scored test windows.
The warm-up is 600 windows. Rolling statistics use a 300-window half-life.

A session is **COMPLETE** only when every grasp label present in that subject's
D1T1 file has repetitions 1, 2, 3, 4, 9, 10, 11, and 12, with at least 10
valid windows for each grasp/repetition pair. Otherwise it is **INCOMPLETE**.
Incomplete sessions are run and reported in their own table but are excluded
from subject means. If D1T1 is incomplete, the subject is recorded and skipped.

## Arms and replay

The fixed arms are `none`, `prefix`, `roll`, `new_only`, `pooled`, `pooled_z`,
`cal_z_only`, and `reference`. `none`, `prefix`, and `roll` use the causal replay
implementations. `reference` is the within-session labeled reference. Calibration
arms use the unchanged calibration model helpers. Each session is run in original
order and with three shuffled seeds for replay; calibration is original order
only. Results are written under `results/multi/<subject>/`.

## Pre-registered hypotheses and safety reporting

- H1: `new_only(k=4)` and `pooled_z(k=4)` beat `none` on accuracy in every
  held-out subject.
- H2: Within each subject, `pooled_z` accuracy rises from `k=1` to `k=4`.
- H3: For the label-free arms `prefix` and `cal_z_only(k=4)`, report the number
  of subjects that beat `none`; no threshold is selected.
- H4: Report rest false-activation percentage, wrong-grasp percentage, and false
  events per minute for every arm. Flag an arm when any subject exceeds twice
  that subject's `none` value.

No significance test, cross-subject pooling, safety claim, or generalization
claim is made.
