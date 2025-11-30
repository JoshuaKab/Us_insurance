## Us_ Insurance Price prediction

In this repository we'll be dealing Us insurance price prediction project we'll go through the process of creating a machine learning algorightme to predict the price of the insurance based on the  customer profile.
We'll use Python and the scikit-learn library to create a model that  can predict the price of the insurance based on the training data.

## Context
This dataset is from kaggle, and it contains the following features:

* **Age**: Customer age of primary beneficiary

* **Gender**: insurance contractor gender, female, male

* **BMI**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height

* **Children**: Number of children covered by health insurance / Number of dependents

* **Smoker**: If the customer Smoking or not

* **Region**: The beneficiary's residential area in the US, northeast, southeast, southwest, northwest
  
## Python Libraries

  The following python libraries was us in this project

  For this lab, we will be using the following libraries:

-   `pandas` for managing the data.
-   `numpy` for mathematical operations.
-  `seaborn` for visualizing the data.
-   `matplotlib` for visualizing the data.
-   `sklearn` for machine learning and machine-learning-pipeline related functions.
  ## Summary

In this notebook, we explored the factors influencing insurance charges using a dataset containing demographic and health-related features. We performed extensive data cleaning, exploratory data analysis (EDA), and feature engineering, including label encoding for categorical variables and outlier removal. Several regression models were trained and evaluated, including Linear Regression, Ridge, Lasso, Polynomial Regression, Random Forest, and XGBoost. Hyperparameter tuning was performed using GridSearchCV, and model performance was assessed using metrics such as RMSE and R² score.

## Conclusion

- **Key Influencers:** Smoking status, age, and BMI were found to be the most significant factors affecting insurance charges, with smoking status showing the strongest positive correlation.
- **Model Performance:** Ensemble models like Random Forest and XGBoost outperformed linear models, achieving higher accuracy on the test set. However, some overfitting was observed, as indicated by better performance on training data compared to test data.
- **Feature Importance:** Feature importance analysis confirmed that 'smoker', 'age', and 'bmi' are the top predictors for insurance charges.

## Recommendations

1. **Business Action:** Insurance companies should pay particular attention to applicants' smoking status, age, and BMI when assessing risk and pricing policies.
2. **Deployment:** The trained model has beem deployed as a web service using Streamlit for real-time insurance charge prediction.

This workflow provides a solid foundation for predictive modeling in insurance pricing 
