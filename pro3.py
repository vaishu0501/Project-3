import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
import joblib

# Load data from Excel file
data = pd.read_excel('data/college_data.xlsx')

# Preprocess data
data.dropna(inplace=True)
data['state'] = data['state'].astype('category')
data['district'] = data['district'].astype('category')
data['college_name'] = data['college_name'].astype('str')
data['course'] = data['course'].astype('category')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('rating', axis=1), data['rating'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer for college name feature
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train['college_name'])
X_test_tfidf = vectorizer.transform(X_test['college_name'])

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer to disk
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Create Flask app
app = Flask(__name__)

# Load model and vectorizer from disk
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from request
    state = request.form['state']
    district = request.form['district']
    college_name = request.form['college_name']
    course = request.form['course']

    # Preprocess user input
    user_input = pd.DataFrame({'state': [state], 'district': [district], 'college_name': [college_name], 'course': [course]})
    user_input_tfidf = vectorizer.transform(user_input['college_name'])

    # Predict best college using trained model
    prediction = model.predict(user_input_tfidf)

    # Fetch real-time reviews from search engine API (assuming search_engine_api is properly implemented)
    try:
        reviews = search_engine_api.fetch_reviews(college_name)
    except Exception as e:
        reviews = {'error': str(e)}  # Handle API errors gracefully

    # Return prediction and reviews to user
    return jsonify({'prediction': prediction.tolist(), 'reviews': reviews})

if __name__ == '__main__':
    app.run(debug=True)
