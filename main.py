from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
import pickle

# Example pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Fit the pipeline on some example data
example_data = pd.DataFrame({
    'beds': [2, 3, 4],
    'baths': [1, 2, 3],
    'size': [1500, 2500, 3500],
    'zip_code': [90001, 90002, 90003]
})
example_target = [300000, 500000, 700000]

pipe.fit(example_data, example_target)

# Save the pipeline
with open('RidgeModel.pkl', 'wb') as f:
    pickle.dump(pipe, f)
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())
    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = int(request.form.get('beds'))
        bathrooms = float(request.form.get('baths'))
        size = float(request.form.get('size'))
        zipcode = int(request.form.get('zip_code'))

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                                  columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Check pipeline steps
        print("Pipeline steps:")
        for name, step in pipe.named_steps.items():
            print(f"{name}: {type(step)}")

        prediction = pipe.predict(input_data)[0]
        return str(prediction)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
