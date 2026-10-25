# EMG hardware-readiness checklist

This is a planning and safety document, not evidence that the current DB6
results transfer to a live device. The repository currently names the BioAmp
EXG Pill as its active hardware, while the proposed future setup mentions
eight MyoWare channels. The exact sensor, analog front end, ADC, reference
scheme, and firmware must be confirmed before implementation.

## Candidate acquisition architecture

1. Eight separate EMG electrode channels feed a medically appropriate,
   battery-powered analog front end.
2. The analog front end feeds a simultaneous-sampling ADC or a documented
   multiplexed ADC with a known per-channel aperture and timestamp.
3. A microcontroller performs packet framing, sequence numbering, timestamping,
   saturation flags, and USB transport. The host performs feature extraction
   and classification.
4. Keep the emergency-stop path independent from the slower classifier. A
   jaw-clench or hardware fault must fail safe and inhibit motion.

The ADC resolution, input range, gain, common-mode range, channel settling
time, and actual sensor output must be measured on the assembled hardware.
Do not assume that a MyoWare output can be connected directly to an arbitrary
ADC or that eight sensors share a safe ground.

## Sampling and analog conditioning

- Target at least 1 kHz per channel only after confirming the useful EMG
  bandwidth and ADC throughput. Record the actual rate in every file.
- Use an analog anti-alias low-pass filter before the ADC. Digital filtering
  cannot remove frequencies already aliased during acquisition.
- Define the high-pass corner, low-pass corner, filter order, gain, and
  saturation behavior from measured hardware response.
- The offline DB6 pipeline uses a causal fourth-order 20--250 Hz Butterworth
  band-pass, then downsamples from 2 kHz to 1 kHz. This is not a specification
  for MyoWare hardware and must not be copied without validating the sensor and
  analog front end.

## Transport and data integrity

Every packet should include:

- monotonically increasing sample or packet sequence number;
- device timestamp and host receipt timestamp;
- channel count, sampling rate, firmware version, and gain configuration;
- saturation, disconnect, and packet-loss flags;
- a checksum or equivalent integrity check.

The host must count missing, duplicated, late, and out-of-order samples. A
missing-sample condition should inhibit actuation or enter a visible
diagnostic state rather than silently resampling corrupted data.

## Power, grounding, electrodes, and isolation

- Use battery power and a tested isolation boundary for any setup connected
  to a person.
- Do not connect a volunteer to an unverified mains-powered circuit, exposed
  USB ground, or improvised electrode wiring.
- Verify electrode placement, skin preparation, lead strain relief, input
  protection, and common-mode limits with a qualified hardware reviewer.
- Check leakage current, fault conditions, enclosure, and connector
  orientation before human testing.
- A software classifier is not a medical safety control. Motion must be
  mechanically limited and have an accessible physical stop.

## Staged bench-testing plan

1. Validate the ADC and packet format with a precision signal source.
2. Measure gain, frequency response, noise floor, channel crosstalk, clipping,
   timestamp drift, and packet-loss recovery.
3. Replay recorded DB6 and synthetic signals through the complete transport
   path; compare samples and features against the offline implementation.
4. Test disconnects, saturated channels, flat channels, NaNs, delayed packets,
   power interruptions, and classifier exceptions. Each must fail safe.
5. Use an electrical test fixture and electrode simulators before any person.
6. Run a supervised, non-actuating pilot with logged data and an independent
   emergency stop only after hardware and ethics approval.
7. Enable low-force actuation with a physical restraint and conservative
   command timeout only after the preceding stages pass.

## Ethics and human-participant gate

Before recruiting volunteers, obtain institutional ethics/IRB approval or the
applicable local equivalent, written informed consent, a risk assessment,
privacy and retention rules, an adverse-event procedure, and a clear
withdrawal process. Explain that NinaPro DB6 is an offline research dataset
and that its results do not establish safety, clinical benefit, or
generalization to MyoWare recordings.

## Readiness decision

The system is not ready for human-volunteer actuation until the sensor and
ADC are identified, isolation and leakage tests pass, transport integrity is
logged, safety interlocks are independently tested, and the relevant ethics
approval is in place.
