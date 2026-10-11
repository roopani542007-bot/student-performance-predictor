import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "student_data.csv"

def main():
    data = pd.read_csv(DATA_FILE)

    features = [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score",
        "sleep_hours",
    ]
    target = "final_score"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("=== Student Performance Predictor ===")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    sample_student = pd.DataFrame([{
        "study_hours": 6,
        "attendance": 90,
        "previous_marks": 78,
        "assignment_score": 85,
        "sleep_hours": 7
    }])

    predicted_score = model.predict(sample_student)[0]
    print(f"Predicted final score: {predicted_score:.2f}/100")

if __name__ == "__main__":
    main()
