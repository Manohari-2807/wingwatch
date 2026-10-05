"""WingWatch logic: data, model, login checks, access rules (no Streamlit code here)."""
import hmac
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGB_NAME = "XGBoost"
except ImportError:  # keeps the app running if xgboost fails to install
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor
    XGB_NAME = "Gradient Boosting (XGBoost stand-in)"

TARGET = "landing_score"
FEATURES = ["flight_hours", "sim_hours", "total_landings", "ifr_landings",
            "approach_speed_consistency", "flare_height_accuracy",
            "touchdown_precision", "crosswind_score", "prev_avg_score"]

FACTORS = [
    ("approach_speed_consistency", "Approach speed consistency", "Stabilized Approaches"),
    ("flare_height_accuracy", "Flare height accuracy", "Flare Technique"),
    ("touchdown_precision", "Touchdown point precision", "Touchdown Point Control"),
    ("crosswind_score", "Crosswind landing score", "Crosswind Landings"),
]
HOURS = [  # key, label, min, max, step
    ("flight_hours", "Flight hours", 0.0, 600.0, 0.25),
    ("sim_hours", "Simulator hours", 0.0, 400.0, 0.25),
    ("total_landings", "Landings", 0, 500, 1),
    ("ifr_landings", "IFR landings", 0, 100, 1),
]
THRESHOLD = 0.60

# Sample records. Replace with your real student data later.
STUDENTS = {
    "student1": dict(name="Student A", flight_hours=250.75, sim_hours=180.5, total_landings=120,
                     ifr_landings=15, approach_speed_consistency=0.80, flare_height_accuracy=0.70,
                     touchdown_precision=0.90, crosswind_score=0.55,
                     history=[62, 66, 65, 71, 74, 78, 80]),
    "student2": dict(name="Student B", flight_hours=60.0, sim_hours=40.0, total_landings=38,
                     ifr_landings=2, approach_speed_consistency=0.50, flare_height_accuracy=0.45,
                     touchdown_precision=0.55, crosswind_score=0.40,
                     history=[40, 44, 43, 50, 52, 55]),
    "student3": dict(name="Student C", flight_hours=410.0, sim_hours=220.0, total_landings=260,
                     ifr_landings=40, approach_speed_consistency=0.92, flare_height_accuracy=0.88,
                     touchdown_precision=0.90, crosswind_score=0.85,
                     history=[78, 82, 84, 83, 88, 90, 91]),
}


# ------------------------------------------------------------ data + model
def make_data(n=1500, seed=7):
    """SYNTHETIC training data. Swap for real logs when you have them."""
    rng = np.random.default_rng(seed)
    fh = rng.gamma(3.0, 40, n).clip(10, 600)
    sh = rng.gamma(2.5, 35, n).clip(0, 400)
    tl = (fh * rng.uniform(0.35, 0.7, n)).round()
    il = (tl * rng.uniform(0.02, 0.2, n)).round()
    skill = rng.beta(4, 2.5, n)
    nz = lambda s: rng.normal(0, s, n)
    df = pd.DataFrame({
        "flight_hours": fh, "sim_hours": sh, "total_landings": tl, "ifr_landings": il,
        "approach_speed_consistency": (skill + nz(.10)).clip(0, 1),
        "flare_height_accuracy": (skill + nz(.12)).clip(0, 1),
        "touchdown_precision": (skill + nz(.12)).clip(0, 1),
        "crosswind_score": (skill * rng.uniform(.6, 1.1, n) + nz(.10)).clip(0, 1),
    })
    df["prev_avg_score"] = (skill * 100 + nz(8)).clip(0, 100)
    df[TARGET] = (28 * df.touchdown_precision + 20 * df.flare_height_accuracy
                  + 18 * df.approach_speed_consistency + 14 * df.crosswind_score
                  + 0.12 * df.prev_avg_score
                  + 6 * np.log1p(df.flight_hours) / np.log1p(600)
                  + 3 * np.log1p(df.sim_hours) / np.log1p(400) + nz(3)).clip(0, 100)
    for col in ["sim_hours", "crosswind_score", "prev_avg_score"]:
        df.loc[rng.random(n) < .04, col] = np.nan      # missing values
    df.loc[rng.random(n) < .01, "flight_hours"] = -1   # bad log entries
    return df


def clean(df):
    df = df[df.flight_hours > 0]
    df = df[df.ifr_landings <= df.total_landings].copy()
    medians = df[FEATURES].median()
    df[FEATURES] = df[FEATURES].fillna(medians)
    return df.dropna(subset=[TARGET]), medians.to_dict()


def train():
    """Returns (forest, medians, comparison_table). Random Forest is the engine."""
    df, medians = clean(make_data())
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    models = {
        "Linear Regression (baseline)": LinearRegression(),
        "Random Forest (main)": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                       random_state=42, n_jobs=-1),
        XGB_NAME: XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        rows.append({"Model": name, "R²": round(float(r2_score(y_te, p)), 3),
                     "MAE (points)": round(float(mean_absolute_error(y_te, p)), 2)})
    return models["Random Forest (main)"], medians, pd.DataFrame(rows)


def predict(forest, medians, inputs, history):
    row = {**inputs, "prev_avg_score": history[-1]}
    row = {f: row.get(f, medians[f]) for f in FEATURES}
    X = pd.DataFrame([row])[FEATURES]
    spread = 1.28 * float(np.std([t.predict(X.values)[0] for t in forest.estimators_]))
    score = float(forest.predict(X)[0])
    clip = lambda v: round(min(max(v, 0.0), 100.0), 1)
    return {"score": clip(score), "low": clip(score - spread), "high": clip(score + spread)}


def category(v):
    return ("Checkride ready" if v >= 85 else "Proficient" if v >= 70
            else "Developing" if v >= 55 else "Needs focused work")


def weak_factors(inputs):
    return sorted([(k, label, focus) for k, label, focus in FACTORS if inputs[k] < THRESHOLD],
                  key=lambda t: inputs[t[0]])


# ------------------------------------------------------------ login + access
def check_login(role, username, password, passwords):
    """passwords: dict like st.secrets['passwords']. Returns a user dict or None."""
    username = (username or "").strip().lower()
    expected = passwords.get(username) if passwords else None
    ok = expected is not None and hmac.compare_digest(str(expected).encode(), (password or "").encode())
    actual_role = "faculty" if username.startswith("faculty") else "student"
    if not ok or actual_role != role:
        return None
    return {"username": username, "role": actual_role}


def visible_students(user):
    """Students get only their own record; faculty get all."""
    if user["role"] == "faculty":
        return dict(STUDENTS)
    return {user["username"]: STUDENTS[user["username"]]} if user["username"] in STUDENTS else {}


def can_adjust_inputs(user):
    return user["role"] == "student"
