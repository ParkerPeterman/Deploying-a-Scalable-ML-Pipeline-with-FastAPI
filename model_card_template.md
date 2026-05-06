# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Algorithm**: Random Forest Classifier.
* **Parameters**: The model uses 100 estimators and a `random_state` of 42 to ensure consistent results during training[cite: 1, 2].
* **Developer**: Developed by Parker Peterman as part of the Data Analytics coursework.

## Intended Use
* **Primary Task**: This model is intended to predict whether an individual's annual income exceeds $50,000 based on census data.
* **Users**: It is designed for researchers or students studying socio-economic trends and classification pipeline development.

## Training Data
* **Dataset**: Census Income Dataset (also known as the "Adult" dataset).
* **Preprocessing**: The training set comprises 80% of the total data. Categorical features (such as education and occupation) were transformed using OneHotEncoding, and the target label was binarized.

## Evaluation Data
* **Split**: The model was evaluated on a held-out test set consisting of 20% of the original dataset.
* **Method**: Performance was assessed using a fixed test split to ensure metrics were derived from data not seen during the training phase.

## Metrics
_Below are the metrics used and the model's performance on the test set:_
* **Precision**: 0.7419
* **Recall**: 0.6384
* **F1-Score**: 0.6863

## Ethical Considerations
* **Biases**: The dataset includes sensitive demographic attributes, including race, sex, and native-country.
* **Fairness**: Because these features are used as inputs, the model may reflect or amplify historical biases present in the 1994 census data. Users should be cautious when applying this model to diverse populations without further fairness testing.

## Caveats and Recommendations
* **Data Age**: The census data is from 1994, which likely does not reflect the current economic or inflationary environment of the 2020s.
* **Improvements**: For better performance, it is recommended to implement hyperparameter tuning or explore more modern algorithms like XGBoost for the classification task.