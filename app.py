from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the dataset and train the model
df = pd.read_csv('train.csv')
df = df[['text', 'label']].dropna()

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Define a function to predict fake or real news
def predict_news(text):
    text_vectorized = tfidf_vectorizer.transform([text])
    prediction = pac.predict(text_vectorized)
    return "Fake" if prediction == 1 else "Real"

# OCR to extract text from image
def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        file_path = 'static/' + file.filename
        file.save(file_path)

        # Extract text from the image using OCR
        extracted_text = ocr_image(file_path)

        # Predict whether the news is fake or real
        result = predict_news(extracted_text)

        return render_template('result.html', result=result, extracted_text=extracted_text)

if __name__ == "__main__":
    app.run(debug=True)
