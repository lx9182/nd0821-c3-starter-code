from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_greeting():
    r = client.get('/')
    assert r.status_code == 200
    print('r: ', r)
    assert r.json() == 'Greeting!'


def test_inference_1():
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
    r = client.post('/inference', json=input)
    assert r.status_code == 200
    assert r.json() == {'salary': '<=50K'}


def test_inference_2():
    input = {
        'age': 50,
        'workclass': 'Private',
        'fnlgt': 193524,
        'education': 'Doctorate',
        'education_num': 16,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Prof-specialty',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 2174,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }
    r = client.post('/inference', json=input)
    assert r.status_code == 200
    assert r.json() == {'salary': '>50K'}
