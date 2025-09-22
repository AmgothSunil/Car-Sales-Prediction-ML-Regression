import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def read_data(path):
    df = pd.read_csv(path)
    return df


def eda(df):
    df = df.drop(['car_ID', 'CarName'], axis=1)
    return df


def train_model(df):
    target = 'price'

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object' and col != target]
    numerical_columns = [col for col in df.columns if col not in categorical_columns and col != target]

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_columns)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='log2',
        max_depth=None,
        criterion='squared_error'
    )

    # Build full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("R2 Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("\n------------------------------")

    joblib.dump(pipeline, "model.joblib")
    print("Pipeline saved as model.joblib")

    return pipeline


# Example Usage
if __name__ == "__main__":
    df = read_data("D:/AI Projects/ml_car_sales/frontend/dataset/CarPrice_Assignment.csv")
    df = eda(df)
    model = train_model(df)
