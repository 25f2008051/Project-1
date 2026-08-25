Fraud Risk Detector

A machine learning model that scores payment transactions for fraud risk and decides whether to **approve**, **challenge with OTP**, or **block** them — built for the AI Risk Manager track (fraud-spike detector).

Strictly defensive: this model only scores and flags transactions. It has no offensive capability — it cannot move funds, retaliate, or take any action beyond approve/challenge/block.

## What it does

1. Generates a synthetic dataset of 5,000 transactions (5% fraud, 95% legit) with realistic overlap between fraud and legit behavior, plus 2% label noise to mimic real-world mislabeled chargebacks.
2. Splits the data into a training set (75%) and a **held-out test set (25%)** that the model never sees during training or threshold tuning.
3. Trains a Gradient Boosting Classifier to predict fraud probability.
4. Picks a decision threshold by minimizing real ₹ cost (missed fraud vs. false alarms), using training data only.
5. Reports honest precision, recall, F1, and cost — measured on the untouched test set.
6. Wraps the model in a `ProductionRiskEngine` that turns a probability into an action (`APPROVE` / `STEP_UP_OTP` / `BLOCK`) with human-readable reason codes for audit logs.
7. Runs two sample transactions through the engine as a sanity check.

## How to run

```bash
pip install numpy pandas scikit-learn
python3 model.py
```

## Example output

```
--- HONEST TEST SET RESULTS (unseen data) ---
Precision:        0.71  (out of everything I flagged, how much was actually fraud)
Recall:           0.75  (out of all the real fraud, how much did I catch)
F1 score:         0.73
Confusion matrix: TP=62  FP=25  FN=21  TN=1142
Estimated cost on test set: Rs 127,819  (missed fraud: 21 cases, false alarms: 25 cases)
```

## Metrics, explained

| Metric | Meaning |
|---|---|
| **Precision** | Of everything flagged as fraud, how much really was fraud. Low precision = too many innocent customers get blocked/challenged. |
| **Recall** | Of all real fraud, how much did the model catch. Low recall = fraud is slipping through. |
| **F1** | Balance between precision and recall in one number. |
| **Cost (₹)** | Missed fraud (`FN × avg fraud amount + ₹1,000 chargeback fine`) + false alarms (`FP × ₹400 friction cost`). This is the number the threshold is actually optimized against — not accuracy. |

## Features used

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `device_age_days` | How long this device has been associated with the account |
| `velocity_1h` | Number of transactions from this account in the last hour |
| `dist_from_home_km` | Distance of the transaction from the user's usual location |
| `is_upi_collect` | Whether the transaction is a UPI collect request (a known scam vector) |

## Decision logic

- `risk_score ≥ block_threshold` → **BLOCK**
- `risk_score ≥ 0.6 × block_threshold` → **STEP_UP_OTP** (challenge, don't outright reject)
- otherwise → **APPROVE**

`block_threshold` is the value found in Step 4 that minimizes total ₹ cost on the training set.

## Known limitations

- **Synthetic data only.** The model has never seen a real transaction. Distributions are hand-picked approximations of fraud vs. legit behavior, not learned from actual BFSI data. Before any real deployment this needs to be retrained on real (or at least a public benchmark, e.g. Kaggle's IEEE-CIS or credit card fraud dataset) transaction history.
- **No concept drift handling.** Fraud patterns change over time (new scam tactics, new devices); this model and threshold are static and would need periodic retraining.
- **Threshold requires periodic retuning.** The cost assumptions (₹1,000 chargeback fine, ₹400 friction cost) are estimates and should be revisited as real cost data becomes available.
- **Reason codes are independent of the model.** They're simple hand-written rules for audit/explainability only — they don't influence the BLOCK/STEP_UP_OTP/APPROVE decision, and haven't been validated against actual fraud outcomes.
- **Single point-in-time split.** A more rigorous evaluation would use cross-validation or a time-based split (train on older transactions, test on newer ones) rather than one random 75/25 split.
- **No automated fund movement or counter-fraud action.** By design, this is detection/scoring only — defensive, not offensive.

## AI usage disclosure

This project was built with AI assistance (Claude). Approximate breakdown of is :
1) sturcture of the model  is  give by me 
2) code is generated using claude

