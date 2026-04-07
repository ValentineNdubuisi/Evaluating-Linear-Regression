# Evaluating-Linear-Regression
## Table of Content
* Aim
* Description
* 1. Data Creation
* 2. Exploratory Data Analysis (EDA) and Data Processing
* 3. Data Visualization
* 4. Implementation of Linear Regression
     

### Aim:
The aim of this project is to investigate and evaluate a statistical technique called Linear Regression alongside its Python Libraries (Scikit-Learn and Statmodels) for data analysis, focusing on their suitability for handling diverse data types and exploring their potential applications in the field of data science.
### Description
This project utilizes a synthetic data constructed to act as an analogy to real-world application in house-price prediction. It has 500 samples and thereby includes the numerical and categorical variables within. The features in this synthetic dataset include:
* Numerical Features: a. House square footage b. Number of bedrooms c. House-age d. Distance to town centre (in km) e. Crime rate per 1,000 people
Categorial Features: a. Location (urban, suburban, and rural) b. Building type (apartment, townhouse, and detached)
Target Variable: a. Price of the house (dependent variable) Various preprocessing methods were therefore applied to the dataset to verify its adequacy for regression modelling. First, missing values in the dataset were checked and imputation techniques were applied wherever appropriate. Next, categorical variables were changed into numerical format using one-hot encoding, which enabled an efficient model interpretation of categorical data. Furthermore, all numerical variables were subjected to different feature scaling methods to standardize their ranges in order to boost the performance of the model. Finally, the dataset was then split into 80% (training data) and 20% (testing data) so that a well-rounded evaluation of the model on unseen data would be possible.
