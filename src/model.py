import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_market_data
from src.indicators import add_indicators


FEATURE_COLUMNS = [
    "SPY_Return_5D",
    "SPY_Return_10D",
    "SPY_Return_20D",
    "SPY_Return_50D",
    "SPY_vs_MA10",
    "SPY_vs_MA20",
    "SPY_vs_MA50",
    "SPY_vs_MA200",
    "SPY_Vol_10D",
    "SPY_Vol_20D",
    "RSI_14",
    "VIX_Level",
    "VIX_Change_5D",
    "VIX_Change_20D",
    "VIX_vs_MA20",
    "QQQ_Return_20D",
    "IWM_Return_20D",
    "QQQ_vs_SPY",
    "IWM_vs_SPY",
    "MA_Bullish",
    "Price_Above_MA50",
    "Price_Above_MA200"
]


def prepare_data():
    df = get_market_data()
    df = add_indicators(df)

    # Create future return target
    df["Future_5D_Return"] = df["SPY"].shift(-5) / df["SPY"] - 1

    # 1 if SPY is higher in 5 days, 0 if lower
    df["Target"] = (df["Future_5D_Return"] > 0).astype(int)

    # Remove missing rows
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

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    latest_features = X.iloc[[-1]]
    latest_probability = model.predict_proba(latest_features)[0][1]

    return model, accuracy, latest_probability, df


if __name__ == "__main__":
    model, accuracy, latest_probability, df = train_model()

    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Latest bullish probability: {latest_probability:.2%}")