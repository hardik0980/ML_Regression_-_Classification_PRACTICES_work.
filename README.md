# machine_Learning_PRACTICES_work

## 1. Support Vector Machine (SVM) for Loan Application Prediction

Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that separates data points into different classes. In this project, we will use SVM to predict loan application status based on applicant details.

**1. Load the Dataset**
Load the dataset into a DataFrame using pandas.

**2. Data Preprocessing**
   
**a>** Handle Missing Values: Identify and fill missing values with appropriate imputation techniques (e.g., mean for numerical columns and mode for categorical columns).

**b>** Encode Categorical Variables: Convert Gender and Loan Status columns into numerical format (e.g., Male = 1, Female = 0; Y = 1, N = 0).

**c>** Feature Selection: Drop the Loan ID column as it does not contribute to the prediction task.

**4. Split the Dataset**
Split the dataset into training (80%) and testing (20%) sets using train_test_split().

**5. Feature Scaling**
Apply StandardScaler() to standardize numerical columns (ApplicantIncome and LoanAmount) for better SVM performance.

**6. Build and Train the SVM Model**
Use the SVC class from sklearn to train the model.
Experiment with different kernels (linear, rbf, etc.) to find the best-performing model.

**7. Evaluate the Model**
**Classification Report:** Generate precision, recall, and F1-score for each class.

**Confusion Matrix:** Plot a heatmap to visualize the classification results.
**Accuracy Score:** Measure the overall accuracy of the model.

**9. Visualize the Decision Boundary**
For datasets with two numerical features, plot the decision boundary to visualize how the SVM classifier separates the classes.




## 2.  Decision Tree || Car Brand Prediction

A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It works by splitting the dataset into smaller subsets based on feature values, eventually forming a tree-like structure with decision nodes and leaf nodes. Below is a step-by-step guide to training and evaluating a decision tree.

**1. Data Preparation**
Load the dataset using pandas.
Handle missing values and encode categorical features.
Split the data into training and testing sets (e.g., 80:20 ratio).

**2. Initialize the Decision Tree Classifier**
Use the DecisionTreeClassifier from sklearn with desired parameters:
criterion: "gini" or "entropy"
splitter: "best" or "random"
max_features: Number of features to consider for the best split.
max_depth: Maximum depth of the tree.

**3. Train the Model**
Fit the classifier on the training data using:

**4. Evaluate the Model**
Test the model on the test data.
Generate metrics like accuracy, precision, recall, and F1-score.

**5. Visualize the Tree**
Use plot_tree from sklearn.tree to visualize the decision tree.

**6. Fine-Tune Parameters**
Experiment with parameters like max_depth, min_samples_split, and ccp_alpha to optimize performance.




## K-Nearest Neighbors (KNN) || Car Prediction

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression tasks. It is one of the simplest and most intuitive algorithms, based on the idea of instance-based learning or lazy learning. In KNN, the prediction for a new data point is made based on the majority class (for classification) or the average of the target values (for regression) of its nearest neighbors in the feature space.

**step 1: Load the Dataset**
Import required libraries (pandas, numpy, etc.).
Load the dataset using pandas.read_csv().

**Step 2: Data Exploration**
Check the dataset structure using head(), info(), and describe().
Identify missing values and data types.
Understand the distribution of the target variable (mpg).

**Step 3: Handle Missing Values**
If there are missing values:
For numerical features, replace with the mean or median.
For categorical features, replace with the mode.

**Step 4: Data Preprocessing**
Convert categorical features (Origin, Model Year) into numerical format using one-hot encoding or label encoding.
Standardize or normalize numerical features (e.g., Displacement, Horsepower, Weight) using StandardScaler or MinMaxScaler.

**Step 5: Split the Dataset**
Divide the dataset into features (X) and target (y), where y = mpg.
Use train_test_split() to split the data into training and testing sets (e.g., 80% train, 20% test).



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

**1. Load the Dataset**
Use pandas to load the "Heart Disease.csv" dataset into a DataFrame for data manipulation.

**2. Handle Missing Values**
Check for null values.
Fill missing values with the column mean to prevent data loss and ensure consistency.

**4. Split the Data**
Divide the data into training (80%) and testing (20%) sets using train_test_split().

**5. Scale Features**
Use StandardScaler() to standardize numerical data, improving model performance and convergence.

**5. Build and Test the Model**
Train a logistic regression model on the training dataset.
Test it on the testing dataset to assess generalization.

**6. Evaluate the Model**
Generate a classification report to review precision, recall, and F1-score.
Plot a confusion matrix as a heatmap for visual clarity.
Calculate the accuracy score to measure overall performance.

**7.Visualize the Decision Boundary**
Plot the decision boundary for binary classes (0 and 1) to illustrate how the model distinguishes between them


