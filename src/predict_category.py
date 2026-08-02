import joblib
import pandas as pd
 
# Load the saved model
model = joblib.load("model/SVM_predict_model.pkl")
 
print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

# User input
while True:
    title = input(" Enter the product title: ")
    if title.lower() == "exit":
        print("Exiting...")
        break
    
    if not title.strip():
        continue
 
    words = title.split()
    word_count = len(words)
    char_count = len(title)
    longest_word = max(len(word) for word in words)
 
    # Create a DataFrame from input
    user_input = pd.DataFrame(
        [
            {
                "product_title": title,
                "title_word_count": word_count,
                "title_char_count": char_count,
                "longest_word_length": longest_word,
            }
        ]
    )
    # Predict sentiment
    prediction = model.predict(user_input)
    print(f"Predicted category: {prediction[0]}\n")