🧠 Deep Learning Practical 02: Adult Income Prediction Using ANN
Project: Binary classification of adult income using an Artificial
Neural Network (ANN)
Target: Predict whether an individual's annual income is >50K or
<=50K.

📌 Project Overview
This practical applies a Multi-Layer Perceptron (MLP) / Artificial
Neural Network to the Adult Income dataset. The workflow covers data
preparation, exploratory analysis, model building, controlled
experiments, performance evaluation, and final model selection.

The target variable is encoded as:

  Income category     Encoded value
  <=50K                       0
>50K                        1

The notebook contains seven tasks, progressing from dataset
preparation to optimization of the final ANN.
🎯 Objectives
- 📂 Load and inspect the Adult Income dataset.
- 🧹 Detect and remove records containing missing-value markers (?).
- 🧾 Remove the specified columns fnlwgt and education.
- 🔢 Encode the target and categorical predictors.
- 📊 Explore income distribution and feature relationships.
- ⚖️ Scale numerical features and create a stratified train/test
  split.
- 🧠 Build and evaluate a baseline ANN.
- 🧪 Compare activation functions, weight initializers, loss
  functions, Batch Normalization configurations, optimizers, and
  learning rates.
- 🏁 Train and evaluate a final optimized ANN.
🗂️ Notebook Tasks
  Task                    Topic                   Main work
  1️⃣                      Data Loading, Cleaning  Dataset inspection,
& EDA                   missing-value handling,
feature removal, target
encoding,
visualizations, one-hot
encoding, scaling, and
stratified split

  2️⃣                      Baseline ANN            Reusable ANN builder,
baseline training,
evaluation, confusion
matrix, classification
report, and early
stopping

  3️⃣                      Activation Functions    Compare ReLU, Tanh,
Sigmoid, and ELU;
inspect validation
performance, ROC
curves, inactive ReLU
units, and gradient
behaviour

  4️⃣                      Weight Initializers     Compare Glorot Uniform,
Glorot Normal, He
Uniform, He Normal, and
zero initialization

  5️⃣                      Loss Functions          Compare Binary
Crossentropy, Mean
Squared Error, Weighted
Binary Crossentropy,
and Focal Loss

  6️⃣                      Batch Normalization     Compare models
with/without Batch
Normalization, test
layer placement, and
inspect gamma/beta
parameters
7️⃣                      Optimizers              Compare SGD, SGD
Momentum, RMSprop,
Adam, and Adam AMSGrad;
evaluate learning-rate
sensitivity and train
the final ANN
🛠️ Technology Stack
- 🐍 Python
- 📓 Jupyter Notebook
- 🧮 NumPy and Pandas --- data handling
- 📈 Matplotlib and Seaborn --- visualizations
- 🤖 Scikit-learn --- preprocessing, data splitting, and
  evaluation metrics
- 🧠 TensorFlow / Keras --- ANN construction and training
📊 Dataset and Preprocessing
The notebook uses the Adult Income dataset (adult.csv) supplied in
a ZIP archive (adult.csv.zip).

Preprocessing performed in the notebook:
1. Strip whitespace from categorical values.
2. Replace ? with missing values and remove rows containing missing
   values.
3. Drop fnlwgt and education, retaining educational-num.
4. Map income labels to binary values.
5. Split data into training and testing subsets using an 80:20 ratio,
   random_state=42, and stratification.
6. Apply one-hot encoding to categorical features and align test
   columns with training columns.
7. Fit StandardScaler on training numerical features and transform
   the test numerical features using the fitted scaler.
The notebook's reported cleaned dataset contains 45,222 records and
uses 80 encoded input features. The later experimental workflow uses
32,559 training samples, 3,618 validation samples, and 9,045
test samples.
🚀 Getting Started
1. Requirements
Use a Python environment with the packages listed in requirements.txt.
2. Place the dataset
Place adult.csv.zip in the same working directory as the notebook. The
notebook also checks /mnt/data/adult.csv.zip as an alternate location.
3. Install dependencies
python -m pip install -r requirements.txt
4. Launch Jupyter
jupyter notebook
Open:

Deep_Learning_Project_2.ipynb
5. Run the notebook
Run the cells from top to bottom, beginning with the imports and
setup. The later tasks depend on variables, preprocessing outputs, and
model-building functions created in earlier tasks.

The notebook creates data/ and plots/ directories and exports
experiment results and trained models during execution.
📈 Final Model Results
The final ANN's reported test-set performance is:

  Metric        Test result
  Accuracy       81.69%
Precision      59.38%
Recall         82.74%
F1-score       0.6914
ROC-AUC        0.9084

The final model has relatively high recall for the >50K class. Its
class-wise test classification report in the notebook reports:

  Class       Precision   Recall   F1-score
  <=50K          0.93     0.81       0.87
>50K           0.59     0.83       0.69

Interpretation: The final model identifies a high proportion of
individuals in the >50K class, while its precision indicates that some
positive predictions are false positives. Model selection in the
notebook emphasizes F1-score and minority-class recall, not accuracy
alone.
ℹ️ Metrics from different tasks may be based on either validation data
or test data. They should not be compared as if they were measured on
the same evaluation split.

📁 Generated Outputs
When the notebook runs successfully, it creates or exports supporting
artifacts, including:
- 📊 PNG visualizations in plots/
- 📄 CSV experiment summaries and metric tables
- 🧪 Processed data and feature-name exports
- 💾 Saved Keras models (.keras)
The final model is saved as:

task7_final_optimized_ann.keras

Exact generated files depend on successful execution of the notebook
cells.
📦 Suggested Project Structure
Deep_Learning_Project_02/
│
├── Deep_Learning_Project_2.ipynb
├── README.md
├── requirements.txt
├── adult.csv.zip
│
├── data/
│   ├── adult.csv
│   ├── processed_adult_income.npz
│   └── feature_names.csv
│
├── plots/
│   └── *.png
│
├── results/
│   └── *.csv
│
└── models/
    └── task7_final_optimized_ann.keras
📌 The notebook creates data/ and plots/. The results/ and
models/ folders above are a suggested organization for collected
exports; move the generated files into them only if you also update
any paths or submission packaging accordingly.

⚠️ Notes
- Run the notebook in order; later tasks depend on earlier task
  variables and outputs.
- Keep the original dataset ZIP available when rerunning the notebook.
- TensorFlow may display hardware/GPU availability warnings depending
  on the operating system. A GPU is not required for this practical.
- Results can vary slightly with library versions, hardware, and
  execution environment.
- Do not treat validation scores and test scores as directly
  interchangeable.
- The notebook's final reported test results are the values documented
  in this README; rerunning training may produce small differences.
👨‍🎓 Academic Practical
Subject: Deep Learning
Practical: 02
Project Title: Adult Income Prediction Using Artificial Neural
Networks
⭐ This project demonstrates an end-to-end ANN classification workflow,
from data preprocessing and experimentation to model evaluation.
