import requests

input = {
    'age': 64,
    'workclass': 'State-gov',
    'fnlgt': 77516,
    'education': 'Bachelors',
    'education_num': 13,
    'marital_status': 'Adm-clerical',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'capital_gain': 2174,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': 'United-States'
}

response = requests.post(
    url='https://salary-prediction-s47r.onrender.com/inference',
    json=input
)

print(response.status_code)
print(response.json())
