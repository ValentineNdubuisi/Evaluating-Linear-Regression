""" 
Linear Regression Technique alongside its Python Libraries for Data Analysis

Aim:
The aim of this project is to investigate and evaluate statisical technique called Linear Regression,
 alongside its Python Libraries (Scikit-Learn and Statmodels) for data analysis,
focusing on their suitability for handling diverse data type and exploring their potential applications in the field of data science.
"""

"""
Description:
This project utilizes a synthetic data constructed to act as an analogy to real-world application in house-price prediction. 
It has 500 samples and thereby includes the numerical and categorical variables within. 
The features in this synthetic dataset include:

 1. Numerical Features: 
 a. House square footage 
 b. Number of bedrooms 
 c. House-age 
 d. Distance to town centre (in km) 
 e. Crime rate per 1,000 people.
 
 2. Categorial Features: 
 a. Location (urban, suburban, and rural) 
 b. Building type (apartment, townhouse, and detached).
 
 3. Target Variable: 
 a. Price of the house (dependent variable) Various preprocessing methods were therefore applied to the dataset to verify its adequacy for regression modelling. 
 First, missing values in the dataset were checked and imputation techniques were applied wherever appropriate. 
 Next, categorical variables were changed into numerical format using one-hot encoding, which enabled an efficient model interpretation of categorical data. 
 Furthermore, all numerical variables were subjected to different feature scaling methods to standardize their ranges in order to boost the performance of the model. 
 Finally, the dataset was then split into 80% (training data) and 20% (testing data) so that a well-rounded evaluation of the model on unseen data would be possible.
"""

"""
1.Data Creation
As said in the description, this data creation process uses Python libraries (NumPy and pandas) 
to generate a synthetic dataset that simulates the characteristics of a realistic housing market for machine learning analysis.

Firstly, numerical features representing house attributes—such as square footage, 
number of bedrooms, house age, distance from the city center, and local crime rate—are generated using random sampling within specified realistic ranges. 
For instance, house sizes are uniformly distributed between 800 and 5,000 square feet, while crime rates vary from 0 to 50 incidents per 1,000 residents.

Secondly, categorical features including 'Location' (Urban, Suburban, Rural) and 'Building_Type' (Apartment, Townhouse, Detached) are randomly assigned to entries, 
simulating diverse housing conditions.

The target variable, house price, is then computed as a function of these numerical and categorical features. 
Specifically, a formula integrates multiple weighted factors: price increases with larger square footage, 
more bedrooms, and newer age, while price decreases with greater distance to the city center and higher crime rates. 
These base calculations are further adjusted by multipliers reflecting the premium or discount associated with specific locations and building types 
(e.g., urban properties or detached houses carry higher coefficients). To reflect natural variability and unpredictability inherent in real housing markets, 
Gaussian noise with zero mean and specified variance is added to the calculated prices.

Finally, the generated data for all features and the computed prices are consolidated into a pandas DataFrame, 
producing a tabular dataset suitable for subsequent statistical analysis and machine learning model development.

This synthetic data generation approach enables controlled experimentation and modeling when access to real housing data is limited or when testing methodologies under varied hypothetical scenarios.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 500

# Numerical Features
square_feet = np.random.randint(800, 5000, num_samples)  # House size
bedrooms = np.random.randint(1, 6, num_samples)  # Number of bedrooms
age = np.random.randint(1, 100, num_samples)  # Age of house
distance_to_city = np.random.uniform(1, 50, num_samples)  # Distance to city center (km)
crime_rate = np.random.uniform(0, 50, num_samples)  # Crime rate per 1000 residents

# Categorical Features
locations = ['Urban', 'Suburban', 'Rural']
building_types = ['Apartment', 'Townhouse', 'Detached']
location = np.random.choice(locations, num_samples)
building_type = np.random.choice(building_types, num_samples)

# Assign price based on features (using a formula for realism)
location_multiplier = {'Urban': 1.8, 'Suburban': 1.3, 'Rural': 0.9}
building_multiplier = {'Apartment': 1.2, 'Townhouse': 1.5, 'Detached': 1.8}

price = (
    (square_feet * 150) + (bedrooms * 12000) + ((100 - age) * 500) -
    (distance_to_city * 2000) - (crime_rate * 1000)
) * np.array([location_multiplier[loc] for loc in location]) * np.array([building_multiplier[b] for b in building_type])

# Add some noise
price += np.random.normal(0, 60000, num_samples)

# Create DataFrame
df = pd.DataFrame({
    'Square_Feet': square_feet,
    'Bedrooms': bedrooms,
    'Age': age,
    'Distance_to_City_Center': distance_to_city,
    'Crime_Rate': crime_rate,
    'Location': location,
    'Building_Type': building_type,
    'Price': price
})

# Save to CSV
df.to_csv(r'C:\Users\USER\Sta_maz\data\house_prices_complete.csv', index=False)

# Show sample data
# Display first few rows of dataset
print("First ten rows of the dataset:")
print(df.head(10))

"""
2.Exploratory Data Analysis (EDA) and Data Preprocessing
Exploratory Data Analysis (EDA) is the mathematical way of analyzing, investigating, 
and summarizing the main characteristics of the synthentic dataset created. 
It often involves a combination of statistical methods and data visualization techniques to provide insights into the structure, 
distribution, and relationships within the data before applying formal modeling or hypothesis testing. 
EDA helps detect patterns, anomalies, and missing values, enabling practitioners to refine data quality and select appropriate analytical methods.

Before we proceed, we need to:
✅ Explore the data with summary statistics
✅ Visualize relationships between features and price
✅ Data Preparation: Handle categorical variables (convert them into a numerical format)
"""

# Importing Libraries 
import matplotlib.pyplot as plt
import seaborn as sns

# Display Summary Statistics
print("\nDataset Summary:")
print(df.describe())

# Set style for plots
sns.set_style("whitegrid")

"""
3.Data Visualization
"""
# Histogram of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()
"""
We use the histogram of the house prices above to look at the price distribution. 
It shows that house prices are highly positively skewed: a major bulk of properties are in the price range of 500K−1M, 
while only a handful of high-value homes caused this skew.
"""

# Scatter Plot: House size Verses Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Square_Feet'], y=df['Price'], hue=df['Location'], alpha=0.7)
plt.title('House Size vs. Price')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.legend(title="Location")
plt.show()
"""
The scatter plot above represents the house size vs. price in colours according to location. 
it indicates the relationship between house size and price, with each dot color-coded according to the location it came from. 
Generally, it would seem that larger homes are more expensive, with the exception of the highest priced being urban homes.
"""

# Box Plot: Prices based on Location
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Location'], y=df['Price'], palette="coolwarm")
plt.title('House Price Variation by Location')
plt.xlabel('Location')
plt.ylabel('Price ($)')
plt.show()
"""
Boxplot for house prices across various locations.
To understand geographical price variation better, the boxplot showes that urban, 
having the highest median price and variability, contrasts sharply with rural, 
which actually has a lower value with more stable pricing.
"""

# Correlation Heat
# First convert categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Location', 'Building_Type'], drop_first=True)

# Display the dataset to confirm the conversion
print(df_encoded)

# Compute correlation on numerical + encoded categorical features
plt.figure(figsize=(8, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
"""
A correlation heatmap of the numerical figures. 
There was a correlation heatmap for identifying important relationships among the features. 
The results exhibited a strong positive correlation between house prices and square footage, 
with an inverse correlation to crime rate and distance to town.
"""

"""
Data Preparation: 
Handle categorical variables. One-Hot Encoding Categorical Features and Splitting Dataset into Training and Test Sets
"""
# Convert the categorical data to numerical data using OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Separate features (X) and target variable (y)
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable

# Identify categorical columns
categorical_cols = ['Location', 'Building_Type']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True).astype(int)  # Avoid dummy variable trap

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display new feature set after encoding
print("\nFeatures after one-hot encoding:")
print(X_train.head())

"""
Train & Evaluate Linear Regression Model (Scikit-learn)
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Linear Regression model
lr_sklearn = LinearRegression()

# Train the model
lr_sklearn.fit(X_train, y_train)

# Make predictions
y_pred_sklearn = lr_sklearn.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred_sklearn)
mse = mean_squared_error(y_test, y_pred_sklearn)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred_sklearn)

# Print evaluation metrics
print("\n🔹 Scikit-learn Linear Regression Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

"""
Train & Evaluate Linear Regression Model (Statsmodels)
"""
import statsmodels.api as sm  

# Identify categorical columns again before encoding
categorical_cols = ['Location', 'Building_Type']  

# Check if categorical columns exist before encoding
existing_categorical_cols = [col for col in categorical_cols if col in X.columns]

# Apply one-hot encoding only if the columns exist
if existing_categorical_cols:
    X = pd.get_dummies(X, columns=existing_categorical_cols, drop_first=True).astype(int)

# Convert dataset to float for Statsmodels compatibility
X_train_sm = X_train.astype(float)
X_test_sm = X_test.astype(float)

# Add intercept column for Statsmodels
X_train_sm = sm.add_constant(X_train_sm)
X_test_sm = sm.add_constant(X_test_sm)

# Train the model
lr_statsmodels = sm.OLS(y_train, X_train_sm).fit()

# Make predictions
y_pred_statsmodels = lr_statsmodels.predict(X_test_sm)

# Print model summary
print("\n🔹 Statsmodels Regression Summary:")
print(lr_statsmodels.summary())

import statsmodels.api as sm

# Add intercept (Statsmodels does not add it by default)
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Train model
lr_statsmodels = sm.OLS(y_train, X_train_sm).fit()

# Make predictions
y_pred_statsmodels = lr_statsmodels.predict(X_test_sm)

# Print model summary
print("\n🔹 Statsmodels Regression Summary:")
print(lr_statsmodels.summary())

# Evaluate model performance
mae_sm = mean_absolute_error(y_test, y_pred_statsmodels)
mse_sm = mean_squared_error(y_test, y_pred_statsmodels)
rmse_sm = mse_sm ** 0.5
r2_sm = r2_score(y_test, y_pred_statsmodels)

# Print evaluation metrics
print("\n🔹 Statsmodels Linear Regression Performance:")
print(f"Mean Absolute Error (MAE): {mae_sm:.2f}")
print(f"Mean Squared Error (MSE): {mse_sm:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_sm:.2f}")
print(f"R-squared (R²): {r2_sm:.4f}")

"""
4. Implementation of Linear Regression
"""
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = r"C:\Users\USER\Sta_maz\data\house_prices_complete.csv"
df = pd.read_csv(file_path)

# Dispalay basic information
print(df.info())
print(df.head())

# Handle categorical data: Encoding 'Location' and 'Building_Type'
encoder = LabelEncoder()
df['Location'] = encoder.fit_transform(df['Location'])
df['Building_Type'] = encoder.fit_transform(df['Building_Type'])

# selecting features and target variables
X = df[['Square_Feet', 'Bedrooms', 'Age', 'Distance_to_City_Center', 'Crime_Rate', 'Location', 'Building_Type']]
y = df['Price']

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
A. Implementation of Scikit-learn
Scikit-learn has smart and straightforward features for performing linear regression by the LinearRegression() function, 
which deals with feature scaling and takes Ordinary Least Squares (OLS) with the use of the models for minimizing the error between the predicted and actual values of the data. 
The use of the model happens when training data is used to train the model and when the house prices are predicted using the test dataset. 
The performances of the models are assessed through the Mean Squared Error (MSE) and R² score for determining accuracy.
"""
# Scikit-learn Implementation
lr_sklearn = LinearRegression()
lr_sklearn.fit(X_train, y_train)
y_pred_sklearn = lr_sklearn.predict(X_test)

# Model Performance
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

print("\nScikit-learn Results:")
print(f"Intercept: {lr_sklearn.intercept_}, Coefficients: {lr_sklearn.coef_}")
print(f"MSE: {mse_sklearn}, R²: {r2_sklearn}")
"""
### Scikit-Learn Result
This section gives the results of Scikit-learn implementation with some interpretation. The evaluation is based on MSE, R² score, and some statistical insights.

Here are some outputs from the Scikit-learn model:

a.Intercept: -272,428.05

b.MSE: 34,697,783,374.37

c.R² Score: 0.869

This R² score is high (86.9%) and represents a good fit for the model; however, it indicates a higher MSE implying that degree predictions will be off to some extent. Feature effect:

a.Square_Feet (298.36) and Bedrooms (33,498.65) have positive effects on prices, while,

b.Age (-1,058.85), Distance to City Center (-4,241.97), and Crime Rate (-2,267.64) have negative effects on prices.

c.Major contributors are Location (269,205.89) and Building Type (95,076.34).

Scikit-learn is efficient in prediction but does not include any indicators of statistical significance.
"""

"""
B. Implementation of Statsmodels
Statsmodels is the best model for providing a precise statistical summary in linear regression. 
It's different from Scikit-learn, and this will add the constant term manually to hold the intercept. 
Since the implementation relies on a method of Ordinary Least Squares (OLS), it also displays additional metrics like p-values, 
confidence levels, and a statistical significance test against each feature.
"""
# Statsmodels Implementation
X_train_sm = sm.add_constant(X_train)  # Add intercept
X_test_sm = sm.add_constant(X_test)
model_sm = sm.OLS(y_train, X_train_sm).fit()
y_pred_sm = model_sm.predict(X_test_sm)

mse_sm = mean_squared_error(y_test, y_pred_sm)
r2_sm = r2_score(y_test, y_pred_sm)

print("\nStatsmodels Results:")
print(model_sm.summary())
"""
Statsmodels Result
This section gives the results of Statsmodels implementation with some interpretation.

Generated from the Statsmodels model:

a. R² Score: 0.851 (slightly lower than Scikit-learn).

b. P-values confirm significant predictors such as Square_Feet, Bedrooms, Age, Crime_Rate, and Location.

Key Observations:

a. The R² score, 85.1 percent, indicates quite strong predictive power.

b. Statistical validation of feature importance involves p-values.

c. The condition number of 1.59e+04 denotes probable multicollinearity.

Statsmodels give a deeper interpretation which is more useful for research and hypothesis testing.
"""

# Visualization: Prediction Vs Actual Price
plt.figure(figsize=(20,15))

# Scikit-Learn plot
plt.subplot(1,1,1)
sns.scatterplot(x=y_test, y=y_pred_sklearn, label="Predicted vs Actual")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Scikit-learn Predictions")
"""
Actual versus Predicted Prices (Scikit-learn):
This scatter diagram above shows actual versus predicted house price using Scikit-learn. 
The R² score is 0.869, which indicates a strong fit with some deviations for larger prices.
"""

# Statsmodels plot
plt.subplot(1,1,1)
sns.scatterplot(x=y_test, y=y_pred_sm, label="Predicted vs Actual")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Statsmodels Predictions")

plt.tight_layout()
plt.show()
"""
Actual versus Predicted Prices (Statsmodels):
The scatterplot shows the actual vs. predicted prices using Statsmodels, 
which report an R² score of 0.851 on account of a strong correlation but some deviations at higher prices.
"""

"""
Discussion
Often, when analyzing this same model among libraries such as Scikit-learn and Statsmodels, 
Scikit-learn was much easier and not requiring preprocessing, while the Statsmodels needed to add an explicit intercept but gave much more detailed statistical output. 
On the other hand, Scikit-learn was efficient, making it convenient to handle large datasets, 
whereas Statsmodel was quite slower yet offered detailed statistical analysis

Different outputs were designed by both. 
Scikit-learn mainly looked into the predictive precision of key indices such as MSE and R²; therefore, statistical significance insights were absent. 
Thus, this need was addressed by bursting p-values and confidence intervals, making it more suitable for potential research and hypothesis testing Each one of these libraries has its own benefits. 
Some examples of libraries include Scikit-learn, which is better for the machine learning task and has good pipeline integration with cross-validation and hyperparameter tuning,
 while Statsmodels is for statistical analysis where researchers can interpret predictor significance and relationships

Scikit-learn lacked p-values that can be used to infer feature importance. 
On the other hand, the indications of multicollinearity in Statsmodels call for pre-processing and refining of data. 
The two models were also said to not account for outliers leading to deviations in predictions.

Statsmodels were somewhat costlier in terms of computation, besides brilliance, 
specifically required feature scaling in certain cases to have a smooth regression performance. 
To sum up, Scikit-learn was the faster, easier tool whereas Statsmodels would give one more statistical insights. 
The decision would stem from whether one needs to predict models or do inference statistically.
"""

"""
Conclusion
This project seeks to test the performance of two libraries for regression analysis: Scikit-learn and Statsmodels, 
focusing on implementation ease, usage efficiency, and statistical insights. In particular, Scikit-learn offered fast, 
easy implementation smartly configured for implementation in large-scale machine learning, while Statsmodels packed much denser statistical interpretation but had to incur more preprocessing.

In speed and prediction performance, Scikit-learn was unmatched; however, Statsmodels beats that in giving deeper analysis using p-values and confidence intervals. 
Problems included multicollinearity, outliers, and feature scaling.

The decision would thus be determined by the analysis' intention:

Scikit-learn is suited for predictive modeling, while for statistical inference, 
it is better to turn to Statsmodels. Both are effective, and the combination could yield preferred results
"""