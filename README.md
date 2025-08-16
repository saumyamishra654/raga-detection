# raga Detection Notebook — README
This notebook takes an audio file → separates stems (vocals / accompaniment) → extracts framewise pitch with SwiftF0 → builds cent/MIDI histograms → finds stable pitch-class peaks (cross-validated across resolutions) → converts peaks to interval patterns → looks up matching ragas in a scale database → scores & ranks raga candidates using melody + accompaniment signals. It is essentially a principled heuristic pipeline that gives you ranked raga hypotheses plus diagnostics (plots, MIDI, CSVs) for human verification.

Table of contents
	1	What this notebook does
	2	Requirements
	3	Quickstart (run this notebook)
	4	Inputs & outputs (where files land)
	5	Pipeline overview (major stages)
	6	Tunables
	7	Issues I run into
	8	Raga DB/CSV format expectations

1) What this notebook does (quick)
	•	Separates an audio file into vocals.wav and accompaniment.wav using Spleeter (2 stems) 
	•	Runs SwiftF0 on each stem to produce a framewise pitch PitchResult (CSV, optional MIDI, note segments).
	•	Builds cent histograms (folded to a single octave) at high and low resolution.
	•	Detects stable peaks (using cross-resolution validation) and maps them to pitch-classes.
	•	Converts pitch-classes → interval sequence → canonical form → looks up matching raga templates in a DB.
	•	Scores candidates using melody mass and accompaniment tonic salience; outputs df_final (ranked candidates) and helpful visualizations.

2) Requirements
Python packages used in the notebook (install with pip if missing):
```python pip install spleeter librosa numpy scipy matplotlib pandas tensorflow swift_f0 ```
Notes:
	•	Getting spleeter to run on my mac was a huge pain: the post I referred to get it working on my device was this: https://github.com/deezer/spleeter/issues/696, once you have a venv running and get the dependency edits sorted, it is smooth sailing from there on as long as the python path is set correctly. I am running 3.10 on my device
	•	The notebook relies on the local swift_f0 implementation (the SwiftF0 wrapper, export_to_csv, segment_notes, etc.), so make sure the pip install above is run from inside this virtual env

3) Quickstart — run the notebook
	1	Edit the audio_path variable near the top of the notebook to point to your .mp3/.wav file
	2	Set instrument_mode =  'sitar' / 'sarod' etc, or just leave it at 'autodetect' to allow all possible tonics
	3	Run cells from top to bottom. The heavy steps (stem separation, pitch detection) will save outputs in separated_stems/<filename>/ so repeated runs are faster
	4	If you want to re-run SwiftF0 detection even when CSVs exist, call analyze_or_load_with_plots(..., force_recompute=True).

4) Inputs & outputs
Inputs:
	•	audio_path (any supported audio file)
	•	raga_list_final.csv — raga DB (see section 9 for format), same as the one in this repo
Outputs (saved under separated_stems/<filename>/)
	•	vocals.wav, accompaniment.wav (from Spleeter)
	•	<prefix>_pitch_data.csv (SwiftF0 framewise CSV)
	•	<prefix>_midi.csv (voiced-frame MIDI values)
	•	<prefix>_pitch.jpg, <prefix>_note_segments.jpg, <prefix>_combined_analysis.jpg (diagnostic plots)
	•	<prefix>_cent_histogram.jpg, <prefix>_frequency_histogram.jpg
In-memory outputs:
	•	pc_cand — list of detected pitch-class candidates (0..11)
	•	final_peak_indices_100 — indices of high-res bins used as final validated peaks
	•	df_final — DataFrame with scoring info and ranked raga candidates

5) Pipeline overview (the main stages)
	1	Stem separation (spleeter:2stems):
	◦	Produces vocals.wav, accompaniment.wav
	◦	Skips separation if separated_stems/<filename>/ exists
	2	Pitch extraction (SwiftF0 wrapped by analyze_or_load_with_plots):
	◦	Loads CSV if present; otherwise runs detection on the audio
	◦	Exports CSV, midi npy/csv, note segmentation, and MIDI file
	◦	Returns PitchResult and arrays of voiced frequencies
	3	Histogram construction:
	◦	Convert voiced frequencies → MIDI → cents folded to 0–1200
	◦	High-res histogram (num_bins_high_res, default 100) and low-res (num_bins_low_res, default 25). Both of these are tunable, I have been using these
	4	Peak detection (robust):
	◦	Smooth high-res histogram with Gaussian filter (circular wrap)
	◦	Use scipy.signal.find_peaks (wrapped to handle circular edge cases) with prominence and distance parameters to find candidate peaks in both resolutions
	◦	Cross-validate high-res peaks with low-res stable peaks using peak_tolerance_cents
	◦	Map validated peaks to nearest semitone centers (±tolerance_cents) and form pc_cand
	5	Intervalization and DB lookup:
	◦	Sort pitch-classes, compute cyclic intervals
	◦	Canonicalize (lexicographically smallest rotation) and lookup in raga_list_final.csv → returns matching ragas for canonical intervals
	6	Scoring & ranking:
	◦	Melody mass on allowed notes, presence score, log-likelihood of note set, complexity penalties, and a tonic salience boost from accompaniment histogram
	◦	Combine using tunable coefficients into fit_score and produce df_final sorted according to fusion and other flags

7) Key params
Tuning these will change sensitivity / strictness:
	•	SwiftF0 thresholds:
	◦	confidence_threshold (vocals: 0.98 — strict; accompaniment: 0.8)
	•	Peak detection:
	◦	num_bins_high_res = 100 (default): bins across the octave
	◦	num_bins_low_res = 25: coarse validation bins
	◦	sigma = 0.9: smoothing sigma for the high-res histogram
	◦	prom_high (default set to max(1.0, 0.03 * max)): peak prominence threshold
	◦	dist_high = 2: minimum separation between high-res peaks (in bins)
	◦	tolerance_cents = 35: how near a peak must be to a semitone center to be accepted
	◦	peak_tolerance_cents = 45: how near a high-res peak must be to a low-res peak to be validated
	•	Scoring weights:
	◦	ALPHA_MATCH = 0.40, BETA_PRESENCE = 0.25, GAMMA_LOGLIKE = 1.0, DELTA_EXTRA = 1.10, COMPLEX_PENALTY = 0.4, TONIC_SALIENCE_WEIGHT = 0.12, SCALE = 1000.0
	◦	I need to figure out how to make these choices more well informed, probably some kinda of regression

8) Issues I ran into that you might:
	•	Spleeter didn’t create stems: check separated_stems/<filename>/ and ensure Spleeter installed + FFmpeg available.
	•	No voiced frames / empty MIDI arrays: ensure detector range is correct, and try lowering conf)idence threshold for poor recording quality (will need to force recompute)
	•	No peaks validated/bad validation: either the histogram is too flat, smoothing/thresholds too strict, or tolerance too narrow. Try lowering prom_high or increasing tolerance_cents.
	•	Fragile notebook cell ordering: many variables are created across cells. If re-running, either restart and run top-to-bottom or wrap pipeline into a function to avoid missing globals.
	•	raga DB format: see next section for expected csv format. If masks are malformed, parsing will skip the row.

10) raga DB / CSV format expectations
The notebook supports two common shapes for each row:
	•	mask column: a string like "1,0,1,0,..." (12 entries) OR
	•	12 separate columns, 0..11, each being 0/1, and optionally names or raga for the raga name.
For each valid row the notebook:
	•	extracts mask_abs as a 12-element binary mask,
	•	builds the set of pitch class indices where the mask is 1,
	•	computes circular intervals and canonicalizes them for lookup.
Example accepted rows (CSV):
raga,mask,note1,... (if using mask string)
raga,mask
Bhupali,1,0,0,0,1,0,0,1,0,0,0,0
or
raga,mask
Bhupali,"1,0,0,0,1,0,0,1,0,0,0,0"
Either form is parsed by the notebook (I had two forms of the csv, so it handles both in case you want to add more ragas)

Example: interpret df_final (what to look for)
	•	raga, tonic: candidate name and Sa index (0..11)
	•	fit_score (scaled): combined fit number — higher is better
	•	primary_score: melody Sa + strongest Ma/Pa boost
	•	salience: accompaniment counts supporting that tonic
	•	match_mass: fraction of melody mass falling on the raga’s allowed notes

