from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')
lr_model = joblib.load(model_path)

# Define the mapping from numeric predictions to species names
species_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data and convert to float
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Prepare input data as a 2D array
        pred_arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        model_prediction = lr_model.predict(pred_arr)
        prediction = int(model_prediction[0])
        
        # Map prediction to species
        species = species_mapping.get(prediction, 'Unknown')
        
        return render_template('predict.html', prediction=species)
    
    except ValueError:
        return render_template('predict.html', error="Please enter valid values for all features.")
    
    except FileNotFoundError:
        return "Model file not found. Please ensure the correct path."
    
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
