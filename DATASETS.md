# Datasets

## EEG + fMRI — Inner Speech (ds006033) ✅ PRIMARY
- Source: openneuro.org/datasets/ds006033
- Size: 7.3GB
- Content: Simultaneous EEG + fMRI during inner speech task
- Location: data/eeg_fmri_innerspeech/
- Files: .nii.gz (fMRI), .eeg (EEG)
- Why: Only public dataset with simultaneous EEG+fMRI during inner speech

## EEG — ZuCo 2.0 (ds002791) ✅ SECONDARY
- Source: openneuro.org/datasets/ds002791
- Size: 10GB
- Content: EEG during natural reading, 18 subjects, 739 sentences
- Location: data/eeg_zuco/
- Files: .eeg, .vhdr, .vmrk
- Status: Mostly complete, retrying timed-out files

## fMRI — Huth Dataset (ds003020) ✅ SECONDARY
- Source: openneuro.org/datasets/ds003020
- Size: 9.5GB
- Content: fMRI during naturalistic story listening, 8 subjects
- Location: data/fmri_huth/
- Files: .nii.gz
- Why: Most cited fMRI narrative language dataset

## MEG — MASC-MEG ✅ COMPLETE
- Size: 18GB
- Subjects: sub-01, sub-02, sub-05, sub-07, sub-08 (5 subjects, 2 sessions each)
- File format: .con (KIT/Yokogawa system) — use mne.io.read_raw_kit()
- Location: ag3kj/osfstorage/

## EEG — EEGMMIDB ✅ BASELINE
- Source: physionet.org/content/eegmmidb/1.0.0/
- Size: 79MB
- Content: Motor imagery EEG, S001+S002
- Location: data/eeg/eegmmidb/
