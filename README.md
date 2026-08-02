# MACHINE_LEARNING_PROJECT
This project is intended for testing machine learning models.

# Product Category Classification using Machine Learning

This project focuses on automated product category classification based on product titles and extracted textual features using Natural Language Processing (NLP) and Machine Learning techniques. The final trained **Support Vector Machine (SVM)** model achieves an accuracy of **97%** in predicting product categories.

---

## 📁 Project Structure
.
├── data/
│   ├── products.csv           # Raw unprocessed dataset
│   └── cleaned_dataset_for_ML_Train.csv   # Preprocessed dataset for ML
├── notebooks/
│   ├── 01_Data_Cleaning_and_Feature_Engineering.ipynb    # Data cleaning and feature engineering
│   └── 02_ML_Model_Training_and_Evaluation.ipynb        # Model evaluation and comparison
├── src/
│   ├── train_model.py                 # Script to train SVM model on full dataset
│   └── predict_category.py         # CLI application for real-time predictions
├── .gitignore                         # Excludes virtual environments and model artifacts (*.pkl)
└── README.md                          # Project documentation


## 📊 Methodology & Model Evaluation

The dataset undergoes preprocessing using a `ColumnTransformer` pipeline combining:
* **TF-IDF Vectorization** on `product_title` text.
* **MinMaxScaler** on extracted numerical features (`title_word_count`, `title_char_count`, `longest_word_length`).

Multiple classification algorithms were trained and evaluated on test data:
   
| Model                                |       Accuracy | Macro F1-Score | Weighted F1-Score |

| **Support Vector Machine (SVM)**     |    **0.97**    | **0.97**       |      **0.97**   |
| **Random Forest**                    |    **0.96**    | **0.96**       |      **0.96**   |
| **Logistic Regression**              |    **0.96**    | **0.96**       |      **0.96**   |
| **Naive Bayes**                      |    **0.93**    | **0.88**       |      **0.92**   |



### Final Model Selection
In training various models, the **Support Vector Machine (SVM)** achieved the best overall performance and accuracy, which is clearly visible in the confusion matrix heatmap visualization. It recorded the highest accuracy at **0.97**. Although a higher accuracy does not inherently imply a better trained model, its superior F1-score and correctly classified categories in the matrix confirm that SVM is indeed the best performing model.

---

## 💡 Interactive Testing & Input Limitations

During interactive testing, the Support Vector Machine (SVM) model demonstrated excellent performance on descriptive titles.
However, minor misclassifications occurred when inputting raw product codes or highly abbreviated names lacking functional keywords (e.g., predicting *Dishwashers* instead of *Fridge Freezers* for inputs like `Smeg SBS8004PO` or `Bosch Serie 4 KGV39VL31G`).

This is an expected limitation of text-based NLP models. Because universal brand names and factory model codes provide weak semantic signals and low numerical feature values, the model relies heavily on explicit descriptive terms (such as *fridge*, *freezer*, or *cooling*). Including raw product codes in training was intentionally avoided to prevent model overfitting. In a production environment, this constraint can be effectively addressed by enriching input text with full product specifications prior to classification.

---

## 🚀 How to Run

### 1. Requirements
Ensure you have the required Python packages installed:

`pip install pandas scikit-learn, joblib`

### 2. Train the Final Model
To train the SVM model on the entire dataset and save the trained model pipeline:

`python src/train_model.py`

### 3. Run Interactive Prediction
To run the interactive prediction script and test custom product titles in real time:

`python src/predict_category.py`



