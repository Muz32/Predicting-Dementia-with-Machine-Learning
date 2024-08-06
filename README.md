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
- **Model Performance:** How do different machine learning models (Deep Neural Networks and Random Forest,) compare in terms of prediction accuracy and robustness?
- **Feature Importance:** Which features (e.g., age, gender, lifestyle factors, medical history) are the most significant predictors for dementia?

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
  
## Initial DNN Model
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

## Optimised DNN Model
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
The optimised model demonstrates outstanding performance, as indicated by the confusion matrix and classification report. The confusion matrix shows only one misclassification, with 129 true negatives, 120 true positives, no false positive and just one false negative. The model achieves a precision of 0.99 and recall of 1.00 for classifying non-dementia cases, and perfect precision (1.00) and near-perfect recall (0.99) for dementia cases. The overall accuracy score is 1.00, indicating the model’s exceptional ability to accurately predict dementia. This high level of performance suggests the model is highly reliable and effective for practical applications in medical diagnostics and research.

## Random Forest Model
The Random Forest (RF) model was also used to compare results on prediction accuracy and robusteness of the DNN model. When the data was trained with the RF model, all false positive and negative results were eliminated in the first run. This is illustrated in the following confusion matrix generated using the RF model:

![Confusion Matrix](./Images/confusion%20matrix-random%20forest%20model.png)

The RF model achieved perfect classification outcomes for both classes, with no false negatives or false positives indicating a perfect accuracy score of 1.0. In comparison, the DNN model, while highly accurate, had one false negative in the optimised model. Although this minor discrepancy may not be significant in many practical applications, in a critical context like medical diagnostics, the Random Forest model's perfect performance is preferable. Overall, both models are highly reliable, but the Random Forest model demonstrates a slightly superior capability in this case.

### Analysis of Feature Importance
The RF model allows extraction of feature importance from the trained model to identify which features have the most significant impact on the prediction outcomes. This analysis helps in understanding the underlying patterns in the data. The horizontal bar chart below depicts the feature importances:

![Feature Importance](./Images/feature%20importances.png)

In the chart, the Feature importance scores indicate the contribution of each feature in making predictions in the RF model. Higher values indicate more significant features. Here is an analysis of the feature importance output:

- **Prescription_None (0.317741)**:
  - This feature has the highest importance, indicating that whether a patient has no prescription is a critical factor in predicting dementia.

- **Cognitive_Test_Scores (0.271577)**:
  - Cognitive test scores are also highly important, as they directly relate to cognitive function, a key indicator of dementia.

- **Dosage in mg (0.215594)**:
  - The dosage of medication in milligrams is the third most important feature, suggesting that medication dosage significantly impacts dementia predictions.

- **Depression_Status_No (0.047734)** and **Depression_Status_Yes (0.043856)**:
  - Depression status, whether "No" or "Yes," is important, reflecting the connection between mental health and dementia.

- **APOE_ε4_Negative (0.020965)**:
  - The APOE ε4 allele is a known genetic risk factor for dementia. Whether a patient is negative for this allele is moderately important.

- **Prescription_Galantamine (0.020038)**, **Prescription_Rivastigmine (0.014388)**, **Prescription_Memantine (0.013713)**, **Prescription_Donepezil (0.009260)**:
  - Specific prescriptions for dementia (Galantamine, Rivastigmine, Memantine, Donepezil) have varying degrees of importance, reflecting their roles in treatment.

- **APOE_ε4_Positive (0.007215)**:
  - Being positive for the APOE ε4 allele is less important than being negative but still contributes to the model.

- **Smoking_Status_Current Smoker (0.006524)**:
  - Current smoking status has a small impact, indicating some connection to dementia risk.

- **Other features**:
  - Many features have very low importance scores, such as BloodOxygenLevel (0.002067), Age (0.001656), BodyTemperature (0.001563), etc. These features contribute minimally to the model's predictions.

### Key Insights

- **Top Contributing Features**: The most important features are related to prescriptions, cognitive test scores, and medication dosage. These factors are critical in the context of dementia and its treatment.

- **Mental and Genetic Health**: Depression status and the APOE ε4 allele, both known to be associated with dementia, have moderate importance.

- **Other Health Metrics**: Traditional health metrics like Alcohol Level, Blood Oxygen Level, and Age have relatively low importance in this model.

- **Lifestyle and Demographics**: Factors like physical activity, education level, and smoking status also show low importance, suggesting that the model relies more on direct indicators of cognitive health and treatment.


The **RF model** relies heavily on specific medical and cognitive indicators to predict dementia. Understanding the importance of these features can guide medical professionals in focusing on critical areas for early detection and treatment of dementia. With a perfect accuracy score of 1.0, the model demonstrates exceptional reliability and effectiveness for practical applications in medical diagnostics and research. The insights from this feature importance analysis highlight the critical factors influencing dementia predictions and reinforce the model's robustness.

## Model Applications
- **Aid in Early Medical Diagnosis**: The models can analyse complex datasets to identify early signs of dementia, potentially before symptoms become apparent. This can lead to earlier interventions and better patient outcomes.
- **Empower At-Risk Patients:** The models enable at-risk patients to take proactive steps in managing their health by identifying and modifying risk factors such as smoking, alcohol intake, obesity, physical inactivity, and poor diet. This empowerment can lead to better health outcomes and potentially delay the onset of dementia.
- **Government Resource Allocation in Disease Management:** By predicting the prevalence and progression of dementia in different populations, the machine learning model can help governments allocate resources more effectively. This includes planning for healthcare services, support programs, and infrastructure needs.
- **Accelerate Research:** Machine learning can expedite research by analysing large volumes of data from clinical studies, genetic research, and other sources. This can lead to new insights into the causes and progression of dementia, and potentially to the development of new treatments or preventive measures.


## Files & Folders
```
Predicting-Dementia-with-Machine-Learning (Root Folder)
├── Dementia Prediction - with Deep Neural Networks.ipynb #Jupyter notebook for the initial deep neural network model.
├── Dementia Prediction - with Deep Neural Networks-Optimised.ipynb #Jupyter notebook for the optimised deep neural network model
├── Dementia Prediction - with Random Forest Model.ipynb #Jupyter notebook for the random forest model.
├── Data
│   ├── dementia_data.db # Sqlite Database file with dementia-related data
│   └── dementia_patients_health_data.csv #CSV file with patient health data.
├── Project Proposal
│   └── Machine Learning Project Proposal.docx #Document outlining the project proposal.
├── Images #Folder containing images used in the project.
│   ├── confusion matrix-initial model.png
│   ├── classification report-initial model.png
│   ├── confusion matrix-optimised model.png
│   ├── classification report-optimised model.png
│   ├──  confusion matrix-random forest model.png
│   ├── feature importances.png
│   └── dataset features.png
├── .gitignore
└── README.md   

