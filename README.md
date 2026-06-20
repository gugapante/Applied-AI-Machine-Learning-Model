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

<p align = "justify">
## Linear Regression
The first algorithm tested was linear regression, choosen because the Boston Housing Dataset shows some linear relationship with the housing prices. For example, house values tend to increase with the number of rooms and decrease as the LSTAT value increases.
</p>

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
The success of this project was measured using several evaluation metrics. The primary goal wsa to achieve an R<sup>2</sup> score above 80%, indicating that the model explains more than 80% of the variation in the data. Mean Absolute Error (MAE) was used to measure the average magnitude of prediction errors, with lower values indicating greater accuracy. Root Mean Squared Error (RMSE) was also used to assess prediction accuracy by calculating the average difference between predicted and actual values, where smaller values represent better performance. additionally, 5-fold cross-validation was conducted to evaluate the model's stability and reliability. In each iteration, 80% of the data was used for training and 20% for testing, allowing performance consistency across different data splits to be assessed.

<p align = "center">
  <img width="495" height="351" alt="image" src="https://github.com/user-attachments/assets/8826de0b-0c32-4df4-8e9e-0ad740d500c1" />
  <br>
  <em align = "center">5 fold Cross-validation</em>
</p>

# Discussion of Results
## Baseline Models with no Tuning
The housing values in the dataset are measured in thousands of dollars, meaning a value of 25 represents $25,000.

<p align = "center">
  <img width="693" height="106" alt="image" src="https://github.com/user-attachments/assets/8278513c-4ea9-400c-ba01-2eace2acc014" />
<br>
  <em align = "center">Table 1: Single run results for models without any tuning</em>
</p>

Initial results using untuned models showed that linear regression performed worse than random forest and gradient boosting, particularly in R<sup>2</sup> and error metrics, suggesting that the dataset contains complex, non-linear relationships.

Linear regression achieved an R<sup>2</sup> score of 0.6157, explaining 61.6% of the variance, with an average predicition error of approximately $3,700. The MLP Regressor performed the worst overall, which was expected without any parameter tuning. 

<p align = "center">
  <img width="692" height="105" alt="image" src="https://github.com/user-attachments/assets/1797c4d5-507b-4095-9df9-c800090b7be8" />
  <br>
  <em align = "center">Table 2: Average cross-validation results for models without any tuning</em>
</p>

Cross-validation results showed improved average performance for both linear regression and random forest, with random forest continuing to outperform the other models. Gradient boosting experienced only a minor decrease in performance, while MLP regressor's results declined significantly, indicating that its initial performance may not have been reliable.

The graphs below illustrate the variation in performance across the different validation folds and how the average scores were calculated.

<p align = "center">
  <img width="497" height="271" alt="image" src="https://github.com/user-attachments/assets/2f250385-ec19-4d3d-963e-e493df972a69" />
  <br>
  <em align = "center">Graph 1: MAE Comparison across 5 runs</em>
</p>

<p align = "center">
  <img width="499" height="272" alt="image" src="https://github.com/user-attachments/assets/7b84a819-5108-44b2-b7d4-e87b7cf807f2" />
  <br
  <em align = "center">Graph 2: RMSE Comparison across 5 runs</em>
</p>

<p align = "center">
  <img width="508" height="271" alt="image" src="https://github.com/user-attachments/assets/7c3c0bf5-4650-4146-9105-1000ad49ee06" />
  <br>
  <em align = "center">Graph 3: R<sup>2</sup> Comparison across 5 runs</em>
</p>

## MLP using Hyperparameter Tuning
I ran a grid search to determine the best hyperparameter tuning to improve the MLP regressor model.

<p align = "center">
  <img width="547" height="289" alt="image" src="https://github.com/user-attachments/assets/dd059d21-4f06-44ce-91c8-4c5f64265383" />
  <br>
  <em align = "center">Figure 11: Best hyperparameter grid search setup</em>
</p>

The following results were deemed as the best for the MLP regressor model as the other models were performing well.

<p align = "center">
  <img width="349" height="152" alt="image" src="https://github.com/user-attachments/assets/81c1cc30-3775-4ab4-a2f5-62498bad4891" />
  <br>
  <em align = "center">Figure 12: Best parameters for MLP Model</em>
</p>

After running the models with the new hyperparameter tuning setup, I obtained the following results:

<p align = "center">
  <img width="691" height="106" alt="image" src="https://github.com/user-attachments/assets/f8db25c6-42f3-4517-9a56-2eaf0f86c1be" />
  <br>
  <em align = "center">Table 3: Single run results with hyperparameter tuning</em>
</p>

<p align = "center">
  <img width="691" height="103" alt="image" src="https://github.com/user-attachments/assets/9d6b01a1-1110-4caa-9873-5f5601ed48e4" />
  <br>
  <em align = "center">Table 4: Average cross-validation results with hyperparameter tuning</em>
</p>

Hyperparameter tuning significantly improved the performance of the MLP model. After tuning, the MLP outperformed the linear regression model across all evaluationmetrics in both the single test run and the cross-validation results. However, it still lagged behind the random forest and gradient boosting models, which achieved lower prediction errors and higher R<sup>2</sup> scores. 

On average, the MLP's MAE remained approximately $1,000 higher than the random forest and gradient boosting models, and its R<sup>2</sup> score did not reach the target of 80%. The accompanying graphs illustrate the variation in performance across the 5 cross-validation runs.

<p align = "center">
  <img width="511" height="265" alt="image" src="https://github.com/user-attachments/assets/bcb7b65d-61a9-46f6-ac52-8bd9d47a4939" />
  <br>
  <em align = "center">Graph 4: MAE best parameters across 5 runs</em>
</p>

<p align = "center">
  <img width="505" height="267" alt="image" src="https://github.com/user-attachments/assets/6bff3bdf-6b3d-4961-b24a-fce55e045973" />
  <br>
  <em align = "center">Graph 5: RMSE best parameters across 5 runs</em>
</p>

<p align = "center">
  <img width="520" height="268" alt="image" src="https://github.com/user-attachments/assets/543c3315-05cd-4488-811e-a280d0ae11d6" />
  <br>
  <em align = "center">Graph 6: R<sup>2</sup> best parameters across 5 runs</em>
</p>

## MLP using best parameters and standard scaler
Gradient boosting was identified as the best performing model, outperforming random forest due to its ability to capture complex, non-linear relationships in the dataset. To further improve the MLP model, a standard scalar was added, as MLP performance is highly sensitive to the scale of input features.

<p align = "center">
  <img width="383" height="149" alt="image" src="https://github.com/user-attachments/assets/b40611c8-fb94-420a-bd47-818da82b18ae" />
  <br>
  <em align = "center">Figure 16: Standard Scalar setup for MLP</em>
</p>

Below are the results for this setof runs.

<p align = "center">
  <img width="690" height="104" alt="image" src="https://github.com/user-attachments/assets/f5959198-afc1-4cb2-bcb6-bf47577818f5" />
  <br>
  <em align = "center">Table 5: Single run results with best parameters and standard scalar</em>
</p>

<p align = "center">
  <img width="692" height="104" alt="image" src="https://github.com/user-attachments/assets/40aa2388-87c8-43bb-81c8-ed3eea8bad3a" />
  <br>
  <em align = "center">Table 6: Average results with best parameters and standard scalar</em>
</p>

Adding a standard scalar significantly improved the MLP model's single run performance, increaseing the R<sup>2</sup> score to 0.8549 (85.5% variance explained) and reducing RMSE by around $1,000. However, cross-validation results did not improve, with the average R<sup>2</sup> decreasing and RMSE increasing to levels similar to linear regression. To address this. the model's hyperparameters were further adjusted through trial and error, resulting in a new set ofoptimised parameters shown in the figure below.


<p align = "center">
  <img width="277" height="188" alt="image" src="https://github.com/user-attachments/assets/ce37fce1-4a56-4e33-bb37-49cc1151e354" />
  <br>
  <em align = "center">Figure 17: New best parameters</em>
</p>

By changing the activator, learning rate and the solver, I was able to improve the results to a more
desirable level.

<p align = "center">
  <img width="691" height="102" alt="image" src="https://github.com/user-attachments/assets/02c92cd1-9f11-4777-a7da-424e517f1db2" />
  <br>
  <em align = "center">Table 7: Single run results with new best parameters</em>
</p>

<p align = "center">
  <img width="691" height="103" alt="image" src="https://github.com/user-attachments/assets/715624eb-4dc9-4d5d-9780-081701315a61" />
  <br>
  <em align = "center">Table 8: Average results with new best parameters</em>
</p>

<p align = "center">
  <img width="511" height="279" alt="image" src="https://github.com/user-attachments/assets/b1a83960-55c4-420e-8e9a-827c823580e4" />
  <br>
  <em align = "center">Graph 7: MAE with new best parameters</em>
</p>

<p align = "center">
 <img width="517" height="264" alt="image" src="https://github.com/user-attachments/assets/042c44b0-dfa9-45a4-9278-42f8f4542e37" />
  <br>
  <em align = "center">Graph 8: RMSE with new best parameters</em>
</p>

<p align = "center">
  <img width="510" height="273" alt="image" src="https://github.com/user-attachments/assets/5271617d-b18d-4a3c-9713-aba527c8712c" />
  <br>
  <em align = "center">Graph 9: R<sup>2</sup> with new best parameters</em>
</p>

Although the MLPmodel did not outperform the gradient boosting model, its performance improved substantially compared to linear regression. The tuned MLP achieved a 27.5% lower MAE, a 28.5% lower RMSE, and a 13.6% higher R<sup>2</sup> score than linear regression. However, it remained behind gradient boosting, with 16.9% igher MAE, a 25.2% higher RMSE, and a 12.1% lower R<sup>2</sup> score. Overall, gradient boosting remained the best performing model for this particular dataset.

# Conclusions and Future Improvements
