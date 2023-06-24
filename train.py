from starter.train_model import prepare_data, train_save_model, val_model
from constants import cat_features

if __name__ == '__main__':
    data_path = 'data/cleaned_census.csv'
    model_path = 'model/model.pkl'

    # Train model
    train_data, test_data = prepare_data(data_path)
    train_save_model(train_data, model_path, cat_features)

    # Test model
    val_model(test_data, model_path, cat_features)
