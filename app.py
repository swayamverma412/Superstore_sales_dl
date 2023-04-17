from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the machine learning model
model = tf.keras.models.load_model('predict.h5')
# model.predict([[88.0,9,'Category_Furniture']])

@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle incoming requests


@app.route('/predict', methods=['POST'])
def predict():

 Profit = float(request.form['profit'])
 Quantity = int(request.form['quantity'])
 Category= request.form['category']
 print(Profit, Quantity, Category)
     # Process the input data
 Category = Category.strip().title()
 data=["Category_Furniture", "Category_Office Supplies", "Category_Technology"]
 try:
    Category=data.index(Category)
 except Exception as e:
    print(e)
    return "Invalid category. Please enter a valid category (Furniture/Office Supplies/Technology)."
 try:
    prediction = model.predict([[Profit, Quantity, Category]])
    # return ""
    prediction=prediction[0][0]
    # return ''
    return render_template('index.html', prediction_text="The predicted sales value is %.2f"%prediction)
 except Exception as e:
    print(e)
    print(Profit, Quantity, Category)
    return "Invalid input. Please enter a valid input."

if __name__ == "__main__":
    app.run(debug=True)