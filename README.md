# Applied-AI-Machine-Learning-Model
This is part of my university module for applied AI where I had to create a regression machine learning model to predict the housing prices given a number of specific parameters from the boston housing dataset.

# Functionality of This System
This system is designed to predict the the prices of property in the Boston area using the Boston Housing Dataset based on input features. These features include socio-economic, environmental, and housing-related data such as the crime rate in the area, average number of bedrooms, proximity to the city centre and the demographic.

This system uses regression based machine learning models to identify relationships between property features and house prices, enabling accurate predicitions of property values for new data. It preprocesses data by handling missing values, scaling features, and encoding categorical data variables before training models such as Linear Regression, Random Forest, or ensemble methods. Model performance is evaluated using metrics like RSME and R<sup>2</sup> to measure predicition accuracy and overall effectiveness.

# What are the Basic Inputs of the System
This system takes features as iputs which describe various characteristics of the residential area:
- CRIM: Crime rate per capita.
- ZN: The Proportion of residential land zoned for plots over 25,000 sq. feet.
- INDUS: Proportion of non-retail business acres.
- CHAS: A binary variable indicating proximity to the Charles River (=1 if located close to
  the river and 0 if otherwise).
- RM: Average number of rooms per house.
- AGE: Proportion of owner-occupied units built prior to the 1940s.
- DIS: Weighted distance to five Boston employment centres.
- RAD: Index of accessibility to orbital motorways.
- TAX: Full value property tax rate per $10,000.
- PTRATIO: Pupil-teacher ratio.
- AAPOP: A measure related to the proportion of African American residents (This is an
  ethical issue and is discussed later in the report).
- LSTAT: The percentage of lower level educated residents.

The main output of the system isthe median value (MEDV) of a property for a given set of input features.

# What are the Real World Problems that this System Solves?
The project aims to address the challenge of accurately estimating house prices using machine learning methods. Although the Boston Housing Dataset is outdated, it provides a useful foundation for evaluating how property features infulence housing prices. The model can help buyers, estate agents, and planners understand housing markets trends, make informed decisions, and identify factors affecting property prices. However, care must be taken to avoid biases, particularly when using sensitive demographic features. Future improvements could include using a more recent and locally relevant dataset such as London or Bighton.

# Details Regarding Algorithm Selection
## General Prerequists for all Algorithms
Before applying the algorithms, the dataset was loaded and cleaned by removing any unnessecary whitespaces. 

<p align = "center">
  <img width="389" height="157" alt="image" src="https://github.com/user-attachments/assets/26c33767-8fb1-44c0-b481-6e74612ebe71" />
</p>

The features and target variables were then defined by removing the MEDV column from the feature set. Since the dataset contained only numerical data, no feauture encoding was required.

<p align = "center">
  <img width="480" height="83" alt="image" src="https://github.com/user-attachments/assets/55b4064e-34dd-4f58-b181-692ed7431ad4" />
</p>

To evaluate the performance, shuffle split cross-validation was used. This method randomly shuffles the dataset and creates five different train-test splits, using 80% of the data for training and 20% for testing in each run. This helps assess how well the model can predict values on unseen data.

<p align = "center">
  <img width="461" height="56" alt="image" src="https://github.com/user-attachments/assets/ce6f46fe-45d4-4a3d-8d30-3f51a04d8d35" />
</p>

A for loop was implemented to automate five runs of the model, with all training and predicition steps contained within the loop. 

<p align = "center">
  <img width="689" height="184" alt="image" src="https://github.com/user-attachments/assets/15ffdc38-2534-4f80-bb8b-dfa273851b79" />
</p>

To store the results from each run, empty arrays were created to record the model's Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R<sup>2</sup> Score.

<p align = "center">
  <img width="689" height="109" alt="image" src="https://github.com/user-attachments/assets/3a95d92c-f5b3-4027-88c3-ff698328d57a" />
</p>

## Linear Regression
The first algorithm tested was linear regression, choosen because the Boston Housing Dataset shows some linear relationship with the housing prices. For example, house values tend to increase with the number of rooms and decrease as the LSTAT value increases.

<p align = "center">
  <img width="690" height="497" alt="image" src="https://github.com/user-attachments/assets/6782dddc-e994-4e11-96ad-c11b9e666cfa" />
</p>

Linear regression is simple, easy to interpret, and well suited to prediciting continuous values such as median house prices. It is also relatively resistant to overfitting, which is beneficial given the dataset's small size of 506 records and 13 features. 

Although larger datasets can improve linear regression models, they may also intorduce more noise and complexity. This algorithm was used as a baseline model against which the performance of other algorithms could be compared.

## Random Forest Regressor
The random forest regressor is a supervised learning algorithm that uses multiple decision trees from random subsets of the data to make predictions. It is effective at handling larger datasets and identifying complex, nonn-linear relationships between features, making it well suited to the Boston Housing Dataset.

Unlike a single decision tree, Random Forest reduces the risk of overfitting by averaging the predictions of many independently created trees. This makes it a more reliable and accurate model for complex regression tasks, particularly when predicting continuous values such as median property values.

## Gradient Boosting Regressor
The gradient boosting regressor was chosen to compare its performance with the random forest regressor, as both are ensemble methods. Unlike random forest, which builds decision trees independently, gradient boosting creates trees sequentially, with each new tree aiming to correct errors made by the previous one. This process helps improve predicition accuracy by reducing residual errors over time.

However, because the trees are built sequentially, gradient boosting is more prone to overfitting, particularly when working with noisy data or using too mny iterations. It alos requires more computational resources, longer training times, and careful hyperparameter tuning.

Despite these challenges, gradient boosting is highly effective at modelling complex, non-linear relationships and often achieves high predictive accuracy. This makes it particularly valuable in fields such as health care, where it can identify subtle patterns in data and improve predicitions for conditions such as heart disease and diabetes.

## Multi-Layer Perceptron (MLP) Regressor
The MLP regressor is a type of neural network consisting of an input layer, one or more hidden layers, and an output layer. It is particularly effective at identifying complex relationships between variables, making it a suitable choice for the Boston Housing Dataset, which contains non-linear interactions between features such as room numbers, crime rates, and property age.

# Measuring if the System is Successful



# Discussion of Results



# Conclusions and Future Improvements
