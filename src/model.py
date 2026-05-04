import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from src.data_loader import get_market_data
from src.indicators import add_indicators


FEATURE_COLUMNS = [
    "SPY_Return_5D",
    "SPY_Return_20D",
    "SPY_vs_MA10",
    "SPY_vs_MA50",
    "SPY_Vol_20D",
    "RSI_14",
    "VIX_Change_5D",
    "QQQ_Return_20D",
    "IWM_Return_20D"
]


def prepare_data():
    df = get_market_data()
    df = add_indicators(df)

    # Create future return target
    df["Future_5D_Return"] = df["SPY"].shift(-5) / df["SPY"] - 1

    # 1 if SPY is higher in 5 days, 0 otherwise
    df["Target"] = (df["Future_5D_Return"] > 0).astype(int)

    # Remove missing rows and any rows with NaN indicators
    df = df.dropna()

    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    return df, X, y


def evaluate_model_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies = []
    aucs = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))

        if len(y_test.unique()) == 2:
            aucs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    return np.mean(accuracies), np.std(accuracies), np.mean(aucs) if aucs else None


def train_model():
    df, X, y = prepare_data()

    # Reserve the most recent row as a holdout for final prediction
    latest_features = X.iloc[[-1]]
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    cv_accuracy, cv_accuracy_std, cv_roc_auc = evaluate_model_cv(model, X_train, y_train)

    model.fit(X_train, y_train)
    latest_probability = model.predict_proba(latest_features)[0][1]

    if cv_roc_auc is not None:
        print(f"Cross-validated AUC: {cv_roc_auc:.3f}")

    print(
        f"Cross-validated accuracy: {cv_accuracy:.2%} "
        f"± {cv_accuracy_std:.2%}"
    )

    return model, cv_accuracy, latest_probability, df


if __name__ == "__main__":
    model, accuracy, latest_probability, df = train_model()

    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Latest bullish probability: {latest_probability:.2%}")