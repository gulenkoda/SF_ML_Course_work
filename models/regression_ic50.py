from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import median_absolute_error, precision_score, r2_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "coursework_data.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"
TARGET_COLUMNS = ["IC50, mM", "CC50, mM", "SI"]
INDEX_COLUMN = "Unnamed: 0"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
LOG_UPPER = 13.0


def load_data():
    return pd.read_excel(DATA_PATH).replace([np.inf, -np.inf], np.nan)


def make_features_and_target(df, target):
    drop_cols = [col for col in TARGET_COLUMNS + [INDEX_COLUMN] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols)
    return X, y


def make_preprocessor(scale):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def log1p_target(y):
    return np.log1p(y)


def expm1_clipped(y):
    return np.expm1(np.clip(y, 0.0, LOG_UPPER))


def regression_models():
    return {
        "DummyRegressor (медиана)": (
            DummyRegressor(strategy="median"),
            {},
        ),
        "Ridge (log-цель)": (
            TransformedTargetRegressor(
                regressor=Pipeline([
                    ("prep", make_preprocessor(scale=True)),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]),
                func=log1p_target,
                inverse_func=expm1_clipped,
            ),
            {"regressor__model__alpha": [1.0, 10.0, 100.0]},
        ),
        "KNN": (
            Pipeline([
                ("prep", make_preprocessor(scale=True)),
                ("model", KNeighborsRegressor(weights="distance")),
            ]),
            {"model__n_neighbors": [5, 15, 25]},
        ),
        "ExtraTrees": (
            Pipeline([
                ("prep", make_preprocessor(scale=False)),
                ("model", ExtraTreesRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            {"model__n_estimators": [200], "model__max_depth": [None, 20]},
        ),
        "Градиентный бустинг": (
            Pipeline([
                ("prep", make_preprocessor(scale=False)),
                ("model", HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                )),
            ]),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]},
        ),
    }


def classification_models():
    return {
        "DummyClassifier": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "Логистическая регрессия": (
            Pipeline([
                ("prep", make_preprocessor(scale=True)),
                ("model", LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]),
            {"model__C": [0.2, 1.0, 5.0]},
        ),
        "KNN": (
            Pipeline([
                ("prep", make_preprocessor(scale=True)),
                ("model", KNeighborsClassifier(weights="distance")),
            ]),
            {"model__n_neighbors": [5, 15, 25]},
        ),
        "ExtraTrees": (
            Pipeline([
                ("prep", make_preprocessor(scale=False)),
                ("model", ExtraTreesClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                )),
            ]),
            {"model__n_estimators": [200], "model__max_depth": [None, 20]},
        ),
        "Градиентный бустинг": (
            Pipeline([
                ("prep", make_preprocessor(scale=False)),
                ("model", HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                )),
            ]),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6]},
        ),
    }


def positive_scores(estimator, X_test, pred):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        span = scores.max() - scores.min()
        return (scores - scores.min()) / (span + 1e-12)
    return pred


def save_metrics(task_name, table):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{task_name}.csv"
    table.to_csv(path, index=False)
    print(f"\nМетрики сохранены: {path}")


TARGET = "IC50, mM"
TASK_NAME = "regression_ic50"


def main():
    df = load_data()
    X, y = make_features_and_target(df, TARGET)
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, (estimator, grid) in regression_models().items():
        search = GridSearchCV(
            estimator,
            grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        pred = np.clip(search.predict(X_test), 0.0, None)
        rows.append({
            "model": name,
            "best_params": search.best_params_,
            "cv_mae": -search.best_score_,
            "test_mae": mean_absolute_error(y_test, pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "test_median_ae": median_absolute_error(y_test, pred),
            "test_r2": r2_score(y_test, pred),
        })

    table = pd.DataFrame(rows).sort_values("test_mae")
    print(f"\nРегрессия для {TARGET}")
    print(table.round(3).to_string(index=False))
    save_metrics(TASK_NAME, table)


if __name__ == "__main__":
    main()
