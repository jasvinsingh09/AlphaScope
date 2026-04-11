from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

    # Predict whether SPY will be up more than 1% over the next 10 trading days
    df["Future_10D_Return"] = df["SPY"].shift(-10) / df["SPY"] - 1
    df["Target"] = (df["Future_10D_Return"] > 0.01).astype(int)

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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    latest_scaled = scaler.transform(X.iloc[[-1]])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        )
    }

    best_model_name = None
    best_accuracy = -1
    best_probability = None
    best_importance = None

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            prob = model.predict_proba(latest_scaled)[0][1]
            importance = abs(model.coef_[0])
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            prob = model.predict_proba(X.iloc[[-1]])[0][1]
            importance = model.feature_importances_

        acc = accuracy_score(y_test, preds)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_probability = prob
            best_importance = importance

    feature_importance_df = (
        __import__("pandas").DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": best_importance
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "bullish_probability": best_probability,
        "data": df,
        "feature_importance": feature_importance_df
    }