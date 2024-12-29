# machine_Learning_PRACTICES_work






## 4. Linear Regression || Housing price Prediction

Load the housing_price.csv dataset to a DataFrame and perform the following tasks:
The housing_price dataset contains all numeric data and the median_house_value column is our target variable, so with help of linear regression build a model that can predict accurate house prices.
Perform the below task and build a model.

1. Load the housing_price dataset into DataFrame
2. Find the null value and drop it, if it is there
3. Split x and y into train and test data set based on test size as 0.2 and random_state as 10
4. Call the LinearRegression model then fit the model using train data
5. Print R2 vallue, coefficient and intercept
6. Compare actual and predicted values.
7.Print the final summary


## 5. Logistic Regression || Heart Disease Prediction 

Logistic Regression is a machine learning algorithm used for classification tasks. It predicts probabilities using a sigmoid function and is widely applied in fields like healthcare and finance. Below is a concise guide to implementing logistic regression.

1. Load the Dataset
Use pandas to load the "Heart Disease.csv" dataset into a DataFrame for data manipulation.

2. Handle Missing Values
Check for null values.
Fill missing values with the column mean to prevent data loss and ensure consistency.

4. Split the Data
Divide the data into training (80%) and testing (20%) sets using train_test_split().

5. Scale Features
Use StandardScaler() to standardize numerical data, improving model performance and convergence.

5. Build and Test the Model
Train a logistic regression model on the training dataset.
Test it on the testing dataset to assess generalization.

6. Evaluate the Model
Generate a classification report to review precision, recall, and F1-score.
Plot a confusion matrix as a heatmap for visual clarity.
Calculate the accuracy score to measure overall performance.

7.Visualize the Decision Boundary
Plot the decision boundary for binary classes (0 and 1) to illustrate how the model distinguishes between them


