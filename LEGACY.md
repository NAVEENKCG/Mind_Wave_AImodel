# Legacy Pipelines — Read Before Using

ORBIT AI's CURRENT production pipeline is:
  train_moabb.py → models/moabb_csp_lda.pkl
  Used by: bridge_bioamp.py, predict_realtime.py, 
  predict_websocket.py, diagnose.py

The following files implement OLDER, now-superseded 
pipelines, kept for historical/reference purposes only. 
Their outputs are NOT compatible with moabb_csp_lda.pkl 
or bridge_bioamp.py:

| File | What it does | Status |
|------|--------------|--------|
| fetch_and_process_openneuro.py | Single-channel Fp1, 8-band TGAM-style features | Legacy |
| quick_process.py | ds002721-only, 10-subject, 8-band extraction | Legacy |
| preprocess.py + collect_data.py | 18-feature engineered pipeline for CNN-LSTM | Legacy |
| fine_tune.py | Fine-tunes the old CNN-LSTM (model.py), includes an incomplete pretrain stub | Legacy |
| evaluate.py | Evaluates best_model.pth (old CNN-LSTM), not moabb_csp_lda.pkl | Legacy |
| model.py | CNN-LSTM architecture, superseded by MOABB/Riemannian pipeline | Legacy |

If reviving any of these for a future experiment, update 
this file to reflect the change.
