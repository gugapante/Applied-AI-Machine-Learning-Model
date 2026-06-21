# Applied-AI-Machine-Learning-Model
<p align = "justify">
This is part of my university module for applied AI where I had to create a regression machine learning model to predict the housing prices given a number of specific parameters from the Boston Housing Dataset.
</p>

# Functionality of This System
<p align = "justify">
This system is designed to predict the the prices of property in the Boston area using the Boston Housing Dataset based on input features. These features include socio-economic, environmental, and housing-related data such as the crime rate in the area, average number of bedrooms, proximity to the city centre and the demographic.
<br>
This system uses regression based machine learning models to identify relationships between property features and house prices, enabling accurate predictions of property values for new data. It preprocesses data by handling missing values, scaling features, and encoding categorical data variables before training models such as Linear Regression, Random Forest, or ensemble methods. Model performance is evaluated using metrics like RSME and R<sup>2</sup> to measure prediction accuracy and overall effectiveness.
</p>

# What are the Basic Inputs of the System
This system takes features as inputs which describe various characteristics of the residential area:
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

<p align = "justify">
The main output of the system is the median value (MEDV) of a property for a given set of input features.
</p>

# What are the Real World Problems that this System Solves?
<p align = "justify">
The project aims to address the challenge of accurately estimating house prices using machine learning methods. Although the Boston Housing Dataset is outdated, it provides a useful foundation for evaluating how property features influence housing prices. The model can help buyers, estate agents, and planners understand housing markets trends, make informed decisions, and identify factors affecting property prices. However, care must be taken to avoid biases, particularly when using sensitive demographic features. Future improvements could include using a more recent and locally relevant dataset such as London or Bighton.
</p>

# Details Regarding Algorithm Selection
## General Prerequists for all Algorithms
<p align = "justify">
Before applying the algorithms, the dataset was loaded and cleaned by removing any unnecessary white spaces. 
</p>

<p align = "center">
  <img width="472" height="192" alt="image" src="https://github.com/user-attachments/assets/5696f637-3257-4072-9bb3-9ae43d680957" />
  <br>
  <em align = "center">Figure 1: Loading dataset and stripping whitespaces</em>
</p>

<p align = "justify">
The features and target variables were then defined by removing the MEDV column from the feature set. Since the dataset contained only numerical data, no feature encoding was required.
</p>

<p align = "center">
  <img width="580" height="101" alt="image" src="https://github.com/user-attachments/assets/908c3639-9eb4-41ca-aafd-aa45a0857e36" />
  <br>
  <em align = "center">Figure 2: Selecting the correct output feature</em>
</p>

<p align = "justify">
To evaluate the performance, shuffle split cross-validation was used. This method randomly shuffles the dataset and creates five different train-test splits, using 80% of the data for training and 20% for testing in each run. This helps assess how well the model can predict values on unseen data.
</p>

<p align = "center">
  <img width="461" height="56" alt="image" src="https://github.com/user-attachments/assets/ce6f46fe-45d4-4a3d-8d30-3f51a04d8d35" />
  <br>
  <em align = "center">Figure 3: Cross-validation using shuffle split</em>
</p>

<p align = "justify">
A for loop was implemented to automate five runs of the model, with all training and prediction steps contained within the loop. 
</p>

<p align = "center">
  <img width="689" height="184" alt="image" src="https://github.com/user-attachments/assets/15ffdc38-2534-4f80-bb8b-dfa273851b79" />
  <br>
  <em align = "center">Figure 4: Initialising the algorithms in the for loop</em>
</p>

<p align = "justify">
To store the results from each run, empty arrays were created to record the model's Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R<sup>2</sup> Score.
</p>

<p align = "center">
  <img width="689" height="109" alt="image" src="https://github.com/user-attachments/assets/3a95d92c-f5b3-4027-88c3-ff698328d57a" />
  <br>
  <em align = "center">Figure 5: Arrays to store data in for every run</em>
</p>

## Linear Regression
<p align = "justify">
The first algorithm tested was linear regression, chosen because the Boston Housing Dataset shows some linear relationship with the housing prices. For example, house values tend to increase with the number of rooms and decrease as the LSTAT value increases.
</p>

<p align = "center">
  <img width="690" height="497" alt="image" src="https://github.com/user-attachments/assets/6782dddc-e994-4e11-96ad-c11b9e666cfa" />
  <br>
  <em align = "center">Figure 6: Correlation coefficient ranges for feature variables</em>
</p>

<p align = "justify">
Linear regression is simple, easy to interpret, and well suited to predicting continuous values such as median house prices. It is also relatively resistant to over fitting, which is beneficial given the dataset's small size of 506 records and 13 features. 
<br>
Although larger datasets can improve linear regression models, they may also introduce more noise and complexity. This algorithm was used as a baseline model against which the performance of other algorithms could be compared.
</p>

## Random Forest Regressor
<p align = "justify">
The random forest regressor is a supervised learning algorithm that uses multiple decision trees from random subsets of the data to make predictions. It is effective at handling larger datasets and identifying complex, non-linear relationships between features, making it well suited to the Boston Housing Dataset.
<br>
Unlike a single decision tree, Random Forest reduces the risk of over fitting by averaging the predictions of many independently created trees. This makes it a more reliable and accurate model for complex regression tasks, particularly when predicting continuous values such as median property values.
</p>

## Gradient Boosting Regressor
<p align = "justify">
The gradient boosting regressor was chosen to compare its performance with the random forest regressor, as both are ensemble methods. Unlike random forest, which builds decision trees independently, gradient boosting creates trees sequentially, with each new tree aiming to correct errors made by the previous one. This process helps improve prediction accuracy by reducing residual errors over time.
<br>
However, because the trees are built sequentially, gradient boosting is more prone to over fitting, particularly when working with noisy data or using too many iterations. It also requires more computational resources, longer training times, and careful hyper parameter tuning.
<br>
Despite these challenges, gradient boosting is highly effective at modelling complex, non-linear relationships and often achieves high predictive accuracy. This makes it particularly valuable in fields such as health care, where it can identify subtle patterns in data and improve predictions for conditions such as heart disease and diabetes.
</p>

## Multi-Layer Perceptron (MLP) Regressor
<p align = "justify">
The MLP regressor is a type of neural network consisting of an input layer, one or more hidden layers, and an output layer. It is particularly effective at identifying complex relationships between variables, making it a suitable choice for the Boston Housing Dataset, which contains non-linear interactions between features such as room numbers, crime rates, and property age.
</p>

# Measuring if the System is Successful
<p align = "justify">
The success of this project was measured using several evaluation metrics. The primary goal was to achieve an R<sup>2</sup> score above 80%, indicating that the model explains more than 80% of the variation in the data. Mean Absolute Error (MAE) was used to measure the average magnitude of prediction errors, with lower values indicating greater accuracy. Root Mean Squared Error (RMSE) was also used to assess prediction accuracy by calculating the average difference between predicted and actual values, where smaller values represent better performance. additionally, 5-fold cross-validation was conducted to evaluate the model's stability and reliability. In each iteration, 80% of the data was used for training and 20% for testing, allowing performance consistency across different data splits to be assessed.
</p>

<p align = "center">
  <img width="495" height="351" alt="image" src="https://github.com/user-attachments/assets/8826de0b-0c32-4df4-8e9e-0ad740d500c1" />
  <br>
  <em align = "center">Figure 7: 5 fold Cross-validation</em>
</p>

# Discussion of Results
## Baseline Models with no Tuning
<p align = "justify">
The housing values in the dataset are measured in thousands of dollars, meaning a value of 25 represents $25,000.
</p>

<p align = "center">
  <img width="693" height="106" alt="image" src="https://github.com/user-attachments/assets/8278513c-4ea9-400c-ba01-2eace2acc014" />
<br>
  <em align = "center">Table 1: Single run results for models without any tuning</em>
</p>

<p align = "justify">
Initial results using untuned models showed that linear regression performed worse than random forest and gradient boosting, particularly in R<sup>2</sup> and error metrics, suggesting that the dataset contains complex, non-linear relationships.
<br>
Linear regression achieved an R<sup>2</sup> score of 0.6157, explaining 61.6% of the variance, with an average prediction error of approximately $3,700. The MLP Regressor performed the worst overall, which was expected without any parameter tuning. 
</p>

<p align = "center">
  <img width="692" height="105" alt="image" src="https://github.com/user-attachments/assets/1797c4d5-507b-4095-9df9-c800090b7be8" />
  <br>
  <em align = "center">Table 2: Average cross-validation results for models without any tuning</em>
</p>

<p align = "justify">
Cross-validation results showed improved average performance for both linear regression and random forest, with random forest continuing to outperform the other models. Gradient boosting experienced only a minor decrease in performance, while MLP regressor's results declined significantly, indicating that its initial performance may not have been reliable.
<br>
The graphs below illustrate the variation in performance across the different validation folds and how the average scores were calculated.
</p>

<p align = "center">
  <img width="596" height="325" alt="image" src="https://github.com/user-attachments/assets/5d631776-fc03-4292-b956-183528e88df1" />
  <br>
  <em align = "center">Graph 1: MAE Comparison across 5 runs</em>
</p>

<p align = "center">
  <img width="616" height="326" alt="image" src="https://github.com/user-attachments/assets/4537697b-3536-4f30-9230-b1191de5837c" />
  <br
  <em align = "center">Graph 2: RMSE Comparison across 5 runs</em>
</p>

<p align = "center">
  <img width="620" height="327" alt="image" src="https://github.com/user-attachments/assets/ca074d8d-42f5-4bb0-9128-e8b7739b7b3f" />
  <br>
  <em align = "center">Graph 3: R<sup>2</sup> Comparison across 5 runs</em>
</p>

## MLP using Hyper Parameter Tuning
<p align = "justify">
I ran a grid search to determine the best hyper parameter tuning to improve the MLP regressor model.
</p>

<p align = "center">
  <img width="547" height="289" alt="image" src="https://github.com/user-attachments/assets/dd059d21-4f06-44ce-91c8-4c5f64265383" />
  <br>
  <em align = "center">Figure 11: Best hyper parameter grid search setup</em>
</p>

<p align = "justify">
The following results were deemed as the best for the MLP regressor model as the other models were performing well.
</p>

<p align = "center">
  <img width="349" height="152" alt="image" src="https://github.com/user-attachments/assets/81c1cc30-3775-4ab4-a2f5-62498bad4891" />
  <br>
  <em align = "center">Figure 12: Best parameters for MLP Model</em>
</p>

<p align = "justify">
After running the models with the new hyper parameter tuning setup, I obtained the following results:
</p>

<p align = "center">
  <img width="691" height="106" alt="image" src="https://github.com/user-attachments/assets/f8db25c6-42f3-4517-9a56-2eaf0f86c1be" />
  <br>
  <em align = "center">Table 3: Single run results with hyper parameter tuning</em>
</p>

<p align = "center">
  <img width="691" height="103" alt="image" src="https://github.com/user-attachments/assets/9d6b01a1-1110-4caa-9873-5f5601ed48e4" />
  <br>
  <em align = "center">Table 4: Average cross-validation results with hyper parameter tuning</em>
</p>

<p align = "justify">
Hyper parameter tuning significantly improved the performance of the MLP model. After tuning, the MLP outperformed the linear regression model across all evaluation metrics in both the single test run and the cross-validation results. However, it still lagged behind the random forest and gradient boosting models, which achieved lower prediction errors and higher R<sup>2</sup> scores. 
<br>
On average, the MLP's MAE remained approximately $1,000 higher than the random forest and gradient boosting models, and its R<sup>2</sup> score did not reach the target of 80%. The accompanying graphs illustrate the variation in performance across the 5 cross-validation runs.
</p>

<p align = "center">
  <img width="635" height="327" alt="image" src="https://github.com/user-attachments/assets/65430224-0475-411a-9247-fc3a4045b97e" />
  <br>
  <em align = "center">Graph 4: MAE best parameters across 5 runs</em>
</p>

<p align = "center">
  <img width="624" height="332" alt="image" src="https://github.com/user-attachments/assets/3470cc07-dd15-4a0b-92c5-12b869657b98" />
  <br>
  <em align = "center">Graph 5: RMSE best parameters across 5 runs</em>
</p>

<p align = "center">
  <img width="629" height="328" alt="image" src="https://github.com/user-attachments/assets/e53bed87-7b1b-4dda-8d50-e886b792a6eb" />
  <br>
  <em align = "center">Graph 6: R<sup>2</sup> best parameters across 5 runs</em>
</p>

## MLP using best parameters and standard scaler
<p align = "justify">
Gradient boosting was identified as the best performing model, outperforming random forest due to its ability to capture complex, non-linear relationships in the dataset. To further improve the MLP model, a standard scalar was added, as MLP performance is highly sensitive to the scale of input features.
</p>

<p align = "center">
  <img width="383" height="149" alt="image" src="https://github.com/user-attachments/assets/b40611c8-fb94-420a-bd47-818da82b18ae" />
  <br>
  <em align = "center">Figure 16: Standard Scalar setup for MLP</em>
</p>

<p align = "justify">
Below are the results for this setof runs.
</p>

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

<p align = "justify">
Adding a standard scalar significantly improved the MLP model's single run performance, increasing the R<sup>2</sup> score to 0.8549 (85.5% variance explained) and reducing RMSE by around $1,000. However, cross-validation results did not improve, with the average R<sup>2</sup> decreasing and RMSE increasing to levels similar to linear regression. To address this. the model's hyper parameters were further adjusted through trial and error, resulting in a new set of optimised parameters shown in the figure below.
</p>

<p align = "center">
  <img width="277" height="188" alt="image" src="https://github.com/user-attachments/assets/ce37fce1-4a56-4e33-bb37-49cc1151e354" />
  <br>
  <em align = "center">Figure 17: New best parameters</em>
</p>

<p align = "justify">
By changing the activator, learning rate and the solver, I was able to improve the results to a more
desirable level.
</p>

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
  <img width="613" height="328" alt="image" src="https://github.com/user-attachments/assets/3deee5a9-7aa3-4f3f-9762-291217f18cc9" />

  <br>
  <em align = "center">Graph 7: MAE with new best parameters</em>
</p>

<p align = "center">
 <img width="626" height="329" alt="image" src="https://github.com/user-attachments/assets/ea8fa7c2-26c4-460c-8f6a-220cd2c6de98" />
  <br>
  <em align = "center">Graph 8: RMSE with new best parameters</em>
</p>

<p align = "center">
  <img width="628" height="327" alt="image" src="https://github.com/user-attachments/assets/3715a08d-133b-4b57-9269-965c296d2983" />
  <br>
  <em align = "center">Graph 9: R<sup>2</sup> with new best parameters</em>
</p>

<p align = "justify">
Although the MLP model did not outperform the gradient boosting model, its performance improved substantially compared to linear regression. The tuned MLP achieved a 27.5% lower MAE, a 28.5% lower RMSE, and a 13.6% higher R<sup>2</sup> score than linear regression. However, it remained behind gradient boosting, with 16.9% higher MAE, a 25.2% higher RMSE, and a 12.1% lower R<sup>2</sup> score. Overall, gradient boosting remained the best performing model for this particular dataset.
</p>

# Conclusions and Future Improvements
<p align = "justify">
The goal was to achieve an average cross-validation score above 80% for the MLP model, but this was not consistently achieved across five folds, even though individual runs sometimes exceeded 80%. To improve performance, further data preprocessing is recommended, such as feature selection and the use of correlation heat maps to identify important variables. Increasing the number of cross-validation folds could also be explored, although this may not necessarily improve results.
<br>
Among the models tested, the gradient boosting regressor performed the best, requiring no parameter tuning and producing the most consistent results.
<br>
The dataset also raises ethical and professional concerns. One feature records the number of African Americans in an area, which could reinforce harmful stereotypes by implying a relationship between race and house prices. Additionally, the dataset excludes properties valued above $50,000, resulting in incomplete and potentially misleading data. This limitation reduces confidence in the accuracy and fairness of the models trained on the dataset.
<br>
This type of machine learning would be interesting to do with sports, such as predicting the winner of a tennis tournament of the winner of the world cup. This would require a massive amount of data such as individual player rankings, expected goals per match, offensive and defensive ability etc. However, that would still be a good next step for this project.
</p>
