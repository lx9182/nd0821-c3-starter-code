from starter.train_model import prepare_data, compute_slice
from constants import cat_features

if __name__ == '__main__':
    data_path = 'data/cleaned_census.csv'
    model_path = 'model/model.pkl'
    slice_output_path = 'slice_output.txt'

    # Train model
    train_data, test_data = prepare_data(data_path)

    # Test model
    compute_slice(test_data, model_path, cat_features, slice_output_path)
