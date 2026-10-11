# Student Performance Predictor

A simple machine learning project that predicts a student's final exam score from study hours, attendance, previous marks, assignment score, and sleep hours.

## Technologies
- Python
- Pandas
- Scikit-learn

## How it works
1. Loads student data from `student_data.csv`.
2. Splits the data into training and testing sets.
3. Trains a Linear Regression model.
4. Displays the model's Mean Absolute Error and R² score.
5. Predicts the final score for a sample student.

## Run the project

```bash
pip install -r requirements.txt
python main.py
```

## Project structure

- `main.py` - main program
- `student_data.csv` - sample dataset
- `requirements.txt` - required Python libraries

## Note
This is an educational/demo project. The dataset is synthetic and the prediction should not be used for real academic decisions.
