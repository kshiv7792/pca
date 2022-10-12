from flask import Flask, render_template, request

import pandas as pd
app = Flask(__name__)

import joblib
# import pickle
# model = joblib.load("Data_prep_DimRed")
model = joblib.load("Data_prep_DimRed")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_excel(f)
        
        # Drop the unwanted features
        data1 = data.drop(["UnivID"], axis = 1)
        
        # Read only numeric data
        num_cols = data1.select_dtypes(exclude = ['object']).columns
        
        # Perform PCA using the saved model
        pca_res = pd.DataFrame(model.transform(data1[num_cols]),columns = ['pc0', 'pc1', 'pc2', 'pc3', 'pc4', 'pc5'])

        return render_template("index.html", Y = pca_res.to_html())

if __name__=='__main__':
    app.run(debug = True)
