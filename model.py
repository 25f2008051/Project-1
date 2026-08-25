import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from dataclasses import dataclass

# ==========================================
# STEP 1: CREATE FAKE (SYNTHETIC) TRANSACTION DATA
# ==========================================
# I don't have real transaction data so I'm generating fake data
# that behaves like real fraud/legit transactions would.
# I made sure some fraud looks kind of "normal" and some normal
# transactions look a bit "risky" so it's not too easy for the model.

def build_training_data(n_samples=5000):
    np.random.seed(42)
    n_fraud = int(n_samples * 0.05)   # 5% of transactions are fraud
    n_legit = n_samples - n_fraud

    legit_df = pd.DataFrame({
        'amount': np.random.exponential(scale=1200, size=n_legit) + 100,
        'device_age_days': np.random.randint(15, 365, size=n_legit),
        'velocity_1h': np.random.poisson(lam=0.7, size=n_legit),
        'dist_from_home_km': np.random.exponential(scale=8, size=n_legit),
        'is_upi_collect': np.random.choice([0, 1], size=n_legit, p=[0.80, 0.20]),
        'is_fraud': 0
    })

    fraud_df = pd.DataFrame({
        'amount': np.random.exponential(scale=5000, size=n_fraud) + 500,
        'device_age_days': np.random.randint(0, 60, size=n_fraud),      # some fraud uses older devices too
        'velocity_1h': np.random.poisson(lam=2.5, size=n_fraud),        # overlaps with legit range now
        'dist_from_home_km': np.random.exponential(scale=80, size=n_fraud),
        'is_upi_collect': np.random.choice([0, 1], size=n_fraud, p=[0.45, 0.55]),
        'is_fraud': 1
    })

    df = pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # flipping 2% of the labels on purpose, since real fraud labels
    # are never 100% accurate anyway (chargebacks get mislabeled etc)
    flip_idx = df.sample(frac=0.02, random_state=1).index
    df.loc[flip_idx, 'is_fraud'] = 1 - df.loc[flip_idx, 'is_fraud']

    return df

print("1. Creating synthetic dataset...")
df = build_training_data()

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

# ==========================================
# STEP 2: SPLIT INTO TRAIN / TEST
# ==========================================
# training on ALL the data and testing on the same data was a mistake
# I made earlier - it made the model look perfect but that's fake.
# splitting it properly so I can actually trust the results.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
print(f"2. Split data: {len(X_train)} training rows, {len(X_test)} test rows")

# ==========================================
# STEP 3: TRAIN THE MODEL (only on the training part)
# ==========================================
print("3. Training model...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# STEP 4: PICK A DECISION THRESHOLD (still using only training data)
# ==========================================
# model gives a probability from 0.0 to 1.0, so I need to decide
# at what point I actually call something "fraud".
# doing this on the training set only, so I don't peek at test data yet.
train_probs = model.predict_proba(X_train)[:, 1]

FN_COST_PER_CASE = df[df.is_fraud == 1]['amount'].mean() + 1000  # missing fraud = losing the money + chargeback fine
FP_COST_PER_CASE = 400                                            # blocking a real customer by mistake

best_threshold, lowest_cost = 0.5, float('inf')
for threshold in np.linspace(0.1, 0.8, 71):
    preds = (train_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
    cost = fn * FN_COST_PER_CASE + fp * FP_COST_PER_CASE
    if cost < lowest_cost:
        lowest_cost, best_threshold = cost, threshold

print(f"4. Best threshold found on training data: {best_threshold:.2f}")

# ==========================================
# STEP 5: EVALUATE ON THE TEST SET (data model has never seen before)
# ==========================================
# this is the actual honest answer to "is my model good or not"
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = f1_score(y_test, test_preds)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
test_cost = fn * FN_COST_PER_CASE + fp * FP_COST_PER_CASE

print("\n--- HONEST TEST SET RESULTS (unseen data) ---")
print(f"Precision:        {precision:.2f}  (out of everything I flagged, how much was actually fraud)")
print(f"Recall:           {recall:.2f}  (out of all the real fraud, how much did I catch)")
print(f"F1 score:         {f1:.2f}")
print(f"Confusion matrix: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"Estimated cost on test set: Rs {test_cost:,.0f}  "
      f"(missed fraud: {fn} cases, false alarms: {fp} cases)")

# ==========================================
# STEP 6: WRAPPING THE MODEL INTO A RISK ENGINE
# ==========================================
# this part turns the raw probability into an actual decision
# (approve / challenge with otp / block) plus reasons for the log
@dataclass
class TransactionEvent:
    txn_id: str
    amount: float
    device_age_days: int
    velocity_1h: int
    dist_from_home_km: float
    is_upi_collect: int

@dataclass
class RiskDecision:
    txn_id: str
    risk_score: float
    action: str          # APPROVE, STEP_UP_OTP, BLOCK
    reason_codes: list

class ProductionRiskEngine:
    def __init__(self, trained_model, block_threshold: float):
        self.model = trained_model
        self.block_thresh = block_threshold
        self.step_up_thresh = block_threshold * 0.6

    def evaluate(self, txn: TransactionEvent) -> RiskDecision:
        features = pd.DataFrame([{
            'amount': txn.amount,
            'device_age_days': txn.device_age_days,
            'velocity_1h': txn.velocity_1h,
            'dist_from_home_km': txn.dist_from_home_km,
            'is_upi_collect': txn.is_upi_collect
        }])
        score = float(self.model.predict_proba(features)[0, 1])

        # these rules don't affect the decision, they're just logged
        # so a human reviewer can see why a txn looked risky
        reasons = []
        if txn.velocity_1h > 3:
            reasons.append("HIGH_VELOCITY_SPIKE")
        if txn.device_age_days < 2:
            reasons.append("NEW_DEVICE_UNTRUSTED")
        if txn.dist_from_home_km > 50:
            reasons.append("GEOGRAPHIC_ANOMALY")
        if txn.is_upi_collect == 1:
            reasons.append("UPI_COLLECT_CHANNEL_RISK")

        if score >= self.block_thresh:
            action = "BLOCK"
        elif score >= self.step_up_thresh:
            action = "STEP_UP_OTP"
        else:
            action = "APPROVE"

        return RiskDecision(txn.txn_id, round(score, 4), action, reasons)

# ==========================================
# STEP 7: TRYING IT OUT ON TWO SAMPLE TRANSACTIONS
# ==========================================
# just checking the engine actually behaves sensibly -
# one normal txn, one that should obviously look like fraud
engine = ProductionRiskEngine(trained_model=model, block_threshold=best_threshold)

tx_normal = TransactionEvent("TXN-1001", amount=450.0, device_age_days=180, velocity_1h=0, dist_from_home_km=2.1, is_upi_collect=0)
tx_fraud = TransactionEvent("TXN-1002", amount=7500.0, device_age_days=1, velocity_1h=5, dist_from_home_km=140.0, is_upi_collect=1)

print("\n--- SAMPLE DECISIONS ---")
for tx in [tx_normal, tx_fraud]:
    res = engine.evaluate(tx)
    print(f"[{res.txn_id}] Action: {res.action} | Score: {res.risk_score} | Reasons: {res.reason_codes}")