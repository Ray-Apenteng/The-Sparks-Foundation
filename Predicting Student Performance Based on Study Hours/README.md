# Predicting Student Performance Based on Study Hours: A Simple Linear Regression Approach

## Introduction

Understanding the factors influencing student performance is essential for enhancing academic outcomes. This project explores the relationship between study hours and student scores using simple linear regression, a statistical method that models the relationship between a dependent variable and an independent variable. Using a dataset of study hours and corresponding scores, we will develop a predictive model to estimate student performance based on study habits, providing insights for educators and students to improve academic results.

<img src="craiyon_012909_The_image_showcases_a_student_studying_in_a_serene_environment__surrounded_by_books_a.png">

## Dataset

The dataset used in this project contains information about the number of study hours and the corresponding scores of students. The dataset is sourced from an online repository and includes the following columns:

- **Hours**: The number of hours a student studied.

- **Scores**: The score a student achieved.

## Project Steps

### Data Loading and Exploration

- Load the dataset.

- Explore the dataset to understand its structure and summary statistics.

- Perform graphical analysis to visualize the relationship between study hours and scores.

### Data Preparation

- Separate the features (independent variable) and target (dependent variable).

- Split the data into training and testing sets.

### Model Training

- Train a simple linear regression model on the training data.

### Model Evaluation

- Evaluate the model using metrics like Mean Absolute Error (MAE) and R-squared.

### Visualization

- Visualize the regression line and the predictions.

- Plot the actual vs predicted scores.

### Prediction

- Use the trained model to predict the score for a new data point (e.g., predicting the score for a student who studies for 9.25 hours/day).

## Requirements

- Python 3.x

- pandas

- numpy

- scikit-learn

- matplotlib

- seaborn

## Usage

1. Clone the repository:

   ````bash

   git clone https://github.com/yourusername/student-performance-prediction.git

   cd student-performance-prediction

   ````

2. Install the required packages:

   ````bash

   pip install -r requirements.txt

   ````

3. Run the project:

   ````bash

   python student_performance_prediction.py

   ````

## Conclusion

In this project, we performed a simple linear regression analysis to predict the percentage score of a student based on the number of hours studied. The model was trained using the training set and evaluated using the test set. The Mean Absolute Error (MAE) and R-squared value indicated that the model has a reasonable accuracy in predicting the scores. Further improvements could be made by collecting more data and possibly exploring more complex models if needed.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The dataset used in this project is sourced from an online repository.

- This project was inspired by common data science exercises and tutorials that explore the relationship between study habits and academic performance.
