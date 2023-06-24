# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from .ml.data import process_data
from .ml.model import train_model, compute_model_metrics, inference


def prepare_data(data_path):
    """
    Get and split data for training and testing

    Inputs
    ------
    data_path: The train data file path

    Returns
    -------
    train, test: splited train and test data
    """
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.20)
    return train, test


def train_save_model(train, model_path, cat_features):
    """
    Train and save model
    Parameters
    ----------
    train: train data
    model_path: where to save model
    cat_features: categorical features

    Returns
    -------

    """
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    joblib.dump((model, encoder, lb), model_path)


def val_model(test_data, model_path, cat_features):
    # Load model
    model, encoder, lb = joblib.load(model_path)

    # Test model
    X_test, y_test, encoder, lb = process_data(
        X=test_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_test)
    # Evaluate model
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-beta: {fbeta}")
    return precision, recall, fbeta


def compute_slice(test_data, model_path, cat_features, slice_output_path):
    # Load model
    model, encoder, lb = joblib.load(model_path)

    # Test model
    with open(slice_output_path, 'w') as file:
        for feature in cat_features:
            for value in test_data[feature].unique():
                X_test, y_test, encoder, lb = process_data(
                    X=test_data[test_data[feature] == value],
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb
                )
                preds = inference(model, X_test)
                # Evaluate model
                precision, recall, fbeta = compute_model_metrics(y_test, preds)
                info = f"{feature}-{value} \
                    Precision: {precision} Recall: {recall} F-beta: {fbeta}"
                print(info)
                file.write(f"{info}\n")


def online_inference(model_path, input, cat_features):
    model, encoder, lb = joblib.load(model_path)
    X_test, _, _, _ = process_data(
        X=pd.DataFrame(data=input.values(), index=input.keys()).T,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(model, X_test)
    return '>50K' if preds[0] else '<=50K'
