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
* Categorial Features: a. Location (urban, suburban, and rural) b. Building type (apartment, townhouse, and detached)
* Target Variable: a. Price of the house (dependent variable) Various preprocessing methods were therefore applied to the dataset to verify its adequacy for regression modelling. First, missing values in the dataset were checked and imputation techniques were applied wherever appropriate. Next, categorical variables were changed into numerical format using one-hot encoding, which enabled an efficient model interpretation of categorical data. Furthermore, all numerical variables were subjected to different feature scaling methods to standardize their ranges in order to boost the performance of the model. Finally, the dataset was then split into 80% (training data) and 20% (testing data) so that a well-rounded evaluation of the model on unseen data would be possible.

#### 1.Data Creation
As said in the description, this data creation process uses Python libraries (NumPy and pandas) to generate a synthetic dataset that simulates the characteristics of a realistic housing market for machine learning analysis.

Firstly, numerical features representing house attributes—such as square footage, number of bedrooms, house age, distance from the city center, and local crime rate—are generated using random sampling within specified realistic ranges. For instance, house sizes are uniformly distributed between 800 and 5,000 square feet, while crime rates vary from 0 to 50 incidents per 1,000 residents.

Secondly, categorical features including 'Location' (Urban, Suburban, Rural) and 'Building_Type' (Apartment, Townhouse, Detached) are randomly assigned to entries, simulating diverse housing conditions.

The target variable, house price, is then computed as a function of these numerical and categorical features. Specifically, a formula integrates multiple weighted factors: price increases with larger square footage, more bedrooms, and newer age, while price decreases with greater distance to the city center and higher crime rates. These base calculations are further adjusted by multipliers reflecting the premium or discount associated with specific locations and building types (e.g., urban properties or detached houses carry higher coefficients). To reflect natural variability and unpredictability inherent in real housing markets, Gaussian noise with zero mean and specified variance is added to the calculated prices.

Finally, the generated data for all features and the computed prices are consolidated into a pandas DataFrame, producing a tabular dataset suitable for subsequent statistical analysis and machine learning model development.

This synthetic data generation approach enables controlled experimentation and modeling when access to real housing data is limited or when testing methodologies under varied hypothetical scenarios.

#### 2.Exploratory Data Analysis (EDA) and Data Preprocessing¶
Exploratory Data Analysis (EDA) is the mathematical way of analyzing, investigating, and summarizing the main characteristics of the synthentic dataset created. It often involves a combination of statistical methods and data visualization techniques to provide insights into the structure, distribution, and relationships within the data before applying formal modeling or hypothesis testing. EDA helps detect patterns, anomalies, and missing values, enabling practitioners to refine data quality and select appropriate analytical methods.

Before we proceed, we need to:
✅ Load the dataset
✅ Explore the data with summary statistics
✅ Visualize relationships between features and price
✅ Data Preparation: Handle categorical variables (convert them into a numerical format)

#### 3.Data Visualization
Histogram Of House Prices: We use the histogram of the house prices to look at the price distribution. It shows that house prices are highly positively skewed: a major bulk of properties are in the price range of 1M, while only a handful of high-value homes caused this skew.
Scatter Plot, House Size Verses Price: The scatter plot arepresents the house size vs. price in colours according to location. it indicates the relationship between house size and price, with each dot color-coded according to the location it came from. Generally, it would seem that larger homes are more expensive, with the exception of the highest priced being urban homes.
