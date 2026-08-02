import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_data_set_for_ML_Train")

# Define features and label
x = df[['product_title','title_word_count','title_char_count','longest_word_length']]
y = df['category_label']

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
     ('title', TfidfVectorizer(), 'product_title'),
      ('numeric', MinMaxScaler(),
       ['title_word_count', 'title_char_count', 'longest_word_length'])
      ]
    )

# Define pipeline with the best model (e.g. LinearSVC())
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])
 
# Train the model on the entire dataset
pipeline.fit(x, y)

# Save the model to a file
joblib.dump(pipeline, "model/SVM_predict_model.pkl")
 
print(" Model trained and saved as 'model/SVM_predict.pkl'")