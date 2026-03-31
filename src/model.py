from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

    # 1 if SPY is higher in 5 days, 0 if lower
    df["Target"] = (df["Future_5D_Return"] > 0).astype(int)

    # Remove missing rows
    df = df.dropna()

    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    return df, X, y


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