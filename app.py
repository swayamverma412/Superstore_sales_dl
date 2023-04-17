from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = tf.keras.models.load_model('predict.h5')


@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle incoming requests


@app.route('/predict', methods=['POST'])
def predict():

 Profit = float(request.form['profit'])
 Quantity = int(request.form['quantity'])
 Category= request.form['category']
    
     # Process the input data
 Category = Category.strip().title()

 if Category not in ['Category_Furniture', 'Category_Office Supplies', 'Category_Technology']:
    return "Invalid category. Please enter a valid category (Furniture/Office Supplies/Technology)."

 
 prediction = model.predict([[Profit, Quantity, Category]])
    
 return render_template('index.html', prediction_text=f"The predicted sales value is {prediction[0]:,.2f}")


if __name__ == "__main__":
    app.run(debug=True)