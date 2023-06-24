import pytest
import pandas as pd
from constants import cat_features
from starter.train_model import prepare_data

data_path = 'data/cleaned_census.csv'


@pytest.fixture()
def data():
    return pd.read_csv(data_path)


def test_column(data):
    assert set(cat_features).issubset(set(list(data.columns.values)))


def test_salary(data):
    actual_salaries = list(data['salary'].unique())
    expect_salaries = ['<=50K', '>50K']
    assert len(actual_salaries) == 2
    assert sorted(actual_salaries) == sorted(expect_salaries)


def test_prepare_data():
    train, test = prepare_data(data_path)
    assert train.shape[0] > 0
    assert train.shape[1] == 12
    assert test.shape[0] > 0
    assert test.shape[1] == 12
