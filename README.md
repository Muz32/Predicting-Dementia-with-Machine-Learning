# Dementia Prediction with Deep Neural Network Models

This project is a submission for the Data Analytics Boot Camp, a collaborative educational initiative by Monash University and EdX.

## Collaborators

- AMIN, Muzaffar
- BUI, Vi
- DO, Tasha
- ESSE, Harun

## Project Overview

With the use of machine learning technologies, we aim to predict the likelihood of an individual being diagnosed with dementia based on demographic, lifestyle, and medical data. The effectiveness of the model would aid in practical applications such as early medical diagnosis, government resource allocation in disease management and research in the area.

## General Research Questions:
- **Prediction Accuracy:** How accurately can we predict the onset of dementia using demographic, lifestyle, and medical data?
- **Feature Importance:** Which features (e.g., age, gender, lifestyle factors, medical history) are the most significant predictors for dementia?
- **Model Performance:** How do different machine learning models (Deep Neural Networks and Random Forest,) compare in terms of prediction accuracy and robustness?

## Data Source

The data for this project was taken from `Kaggle.com` under the title `Dementia Patient Health, Prescriptions ML Dataset`, which can be accessed [here](https://www.kaggle.com/datasets/kaggler2412/dementia-patient-health-and-prescriptions-dataset).

License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Dataset Features
The following diagram categorises the dataset features used in the analysis into **binary**, **continuous**, and **categorical** variables:

![Features](./Images/dataset%20features.png)

The description of dataset features is available at the data source link provided earliar.  

## Model Building Steps

### Step 1- Data Preprocessing
- **Loading Data**: The dataset was loaded from an SQLite database.
- **Cleaning Data**: Missing values in relevant columns were filled with 'none' or '0'.
- **Encoding Categorical Variables**: Categorical columns were one-hot encoded into binary columns.
- **Setting Target Variable**: The target variable was set to the "Dementia" column, with the remaining features serving as input variables.
- **Stratified Split**: The data was split into training and testing sets using a stratified split to maintain the same class distribution as the original dataset, ensuring reliable model training and evaluation.
- **Scaling Data**: The data was scaled using a standard scaler.

### Step 2- Define, Compile and Train the Model
- **Model Architecture**: A Deep Neural Network (DNN) was defined with various neurons, layers, and activation functions. Details are provided in the next section.

### Step 3- Model Evaluation
- **Performance Metrics**: A confusion matrix and classification report were generated to evaluate model performance. Further discussion on model performance is provided in subsequent sections.
- **Initial Model**: The initial model was built using the Jupyter Notebook file `Dementia Prediction - Deep Neural Networks`.

### Step 4- Model Optimisation
- **Optimised Model**: An optimised model was developed to improve accuracy. This was done in a separate Jupyter Notebook file `Dementia Prediction - Deep Neural Networks - Optimised`.
- **Comparison with Random Forest**: A Random Forest model was also used to compare performance with the DNN model.
  
## Initial Model
### Neurons, Layers, and Activation Functions Used
- The model was defined using two hidden layers and one output layer
    - Input layer: 45 features
    - First hidden layer: 5 neurons with Rectified Linear Unit (ReLU) activation function
    - Second hidden layer: 2 neurons with ReLU activation function
    - Output layer: 1 neuron with Sigmoid activation function
- The model was compiled using:
    - Optimizer: Adam
    - Loss function: binary_crossentropy
    - Evaluation metrics: accuracy
    - Trained using scaled data and Epocs of 15

The above neurons, layers, and activation functions were selected to balance model complexity and performance, ensuring the model could capture non-linear relationships in the data while avoiding overfitting. They were set as an appropriate benchmark to achieve the desired performance.

### Interpretation of results from Initial Model
#### Confusion Matrix
The chart below summarises the performance of the initial model based on testing data:
![Confusion Matrix](./Images/confusion%20matrix-initial%20model.png)

The following can be interpreted from the confusion matrix:
- **True Positives (115)**: 
  - These are cases where the actual class was Yes and the predicted class was also Yes.
  - The model correctly identified 115 individuals with dementia.
- **True Negatives (128)**: 
  - These are cases where the actual class was No and the predicted class was also No.
  - The model correctly identified 128 individuals without dementia.
- **False Positives (1)**: 
  - These are cases where the actual class was No but the predicted class was Yes.
  - The model incorrectly identified 1 individual as having dementia when they did not.
- **False Negatives (6)**: 
  - These are cases where the actual class was Yes but the predicted class was No.
  - The model incorrectly identified 6 individuals as not having dementia when they actually did.

### Classification Report 
The table below depicts the classification report:

![Classification report](./Images/classification%20report-initial%20model.png)

The following are key insights from the above table:

**Precision:**
- Precision measures the accuracy of the positive predictions. For Class 0, 96% of the predicted negatives are correct, and for Class 1, 99% of the predicted positives are correct.
  
**Recall:**
- Recall measures the ability to find all the relevant cases within a dataset. For Class 0, 99% of the actual negatives are correctly identified, and for Class 1, 95% of the actual positives are correctly identified.

**F1-Score:**
- The F1-score is the harmonic mean of precision and recall, providing a balance between the two. Both classes have a high F1-score of 0.97, indicating a well-performing model.
  
**Support:**
- Support is the number of actual occurrences of each class in the dataset.
  
**Accuracy:**
- Overall, the model correctly predicts 97% of the cases.
  
### Overall Model Performance
The model demonstrates excellent performance metrics with an accuracy rate of 97%, suggesting it is highly effective in predicting dementia. It achieves high precision and recall across both classes, resulting in a robust and reliable model suitable for practical application in medical diagnostics and research. However, there is still room for improvement to address the remaining false negatives and false positives predicted from the testing data. Therefore, a second attempt was made to further improve accuracy and optimise the model.

## Optimised Model:
### Neurons, Layers, and Activation Functions Used
- This time model was defined using three hidden layers and one output layer
  - Input layer: 45 features
  - First hidden layer: 8 neurons with Rectified Linear Unit (ReLU) activation function
  - Second hidden layer: 5 neurons with ReLU activation function
  - Third hidden layer: 2 neurons with ReLU activation function
  - Output layer: 1 neuron with Sigmoid activation function
- The same metrics was used to compile the model again:
  - Optimiser: Adam
  - Loss function: binary_crossentropy
  - Evaluation metrics: accuracy
- Using scaled training data, the Epocs was increased to 25.

### Results from the Optimised Model
The following chart demonstartes the output from the confusion matrix:

![Confusion Matrix](./Images/confusion%20matrix-optimised%20model.png)

The following table outlines the scores from the classification report:

![Classification report](./Images/classification%20report-optimised%20model.png)

### Optimised Model Performance
The optimized model demonstrates outstanding performance, as indicated by the confusion matrix and classification report. The confusion matrix shows only one misclassification, with 129 true negatives, 120 true positives, no false positive and just one false negative. The model achieves a precision of 0.99 and recall of 1.00 for classifying non-dementia cases, and perfect precision (1.00) and near-perfect recall (0.99) for dementia cases. The overall accuracy score is 1.00, indicating the model’s exceptional ability to accurately predict dementia. This high level of performance suggests the model is highly reliable and effective for practical applications in medical diagnostics and research.

## Files & Folders
```
Predicting-Dementia-with-Machine-Learning (Root Folder)
├── Dementia Prediction - with Deep Neural Networks.ipynb #Jupyter notebook for the initial deep neural network model.
├── Dementia Prediction - with Deep Neural Networks-Optimised.ipynb #Jupyter notebook for the optimized deep neural network model
├── Dementia Prediction - with Random Forest Model.ipynb #Jupyter notebook for the random forest model.
├── Data
│   ├── dementia_data.db # Sqlite Database file with dementia-related data
│   └── dementia_patients_health_data.csv # CSV file with patient health data.
├── Project Proposal
│   └── Machine Learning Project Proposal.docx # Document outlining the project proposal.
├── Images # Folder containing images used in the project.
│   ├── confusion matrix-initial model.png
│   ├── classification report-initial model.png
│   ├── confusion matrix-optimised model.png
│   ├── classification report-optimised model.png
│   └── dataset features.png
├── .gitignore
└── README.md
          

              



