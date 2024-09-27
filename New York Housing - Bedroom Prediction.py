import numpy as np
import pandas as pd

# Function to load a CSV file into a DataFrame
def load_csv(file_path):
    return pd.read_csv(file_path)

# Function to perform one-hot encoding on specified categorical columns
def one_hot_encode(data):
    categorical_columns = ['TYPE']
    return pd.get_dummies(data, columns=categorical_columns)

# Function to apply the softmax function to an array
def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Function to normalize numerical columns in a DataFrame
def normalize(data, numerical_columns):
    for col in numerical_columns:
        if col in data.columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
    return data

# Function to initialize weights and biases for a neural network
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    std_dev1 = np.sqrt(2. / input_size)
    std_dev2 = np.sqrt(2. / hidden_size1)
    std_dev3 = np.sqrt(2. / hidden_size2)
    parameters = {
        'W1': np.random.randn(input_size, hidden_size1) * std_dev1,
        'b1': np.zeros((1, hidden_size1)),
        'W2': np.random.randn(hidden_size1, hidden_size2) * std_dev2,
        'b2': np.zeros((1, hidden_size2)),
        'W3': np.random.randn(hidden_size2, output_size) * std_dev3,
        'b3': np.zeros((1, output_size))
    }
    return parameters

# Function to convert labels to one-hot encoding
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y.T

# Function for forward propagation in neural network
def forward_propagation(X, parameters, keep_prob=0.8):
    np.random.seed(0)  # Ensure reproducibility
    W1, b1, W2, b2, W3, b3 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2'], parameters['W3'], parameters['b3']
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU activation
    D1 = (np.random.rand(*A1.shape) < keep_prob) / keep_prob  # Dropout mask
    A1 *= D1  # Apply dropout

    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)  # ReLU activation
    D2 = (np.random.rand(*A2.shape) < keep_prob) / keep_prob  # Dropout mask
    A2 *= D2  # Apply dropout

    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)  # Softmax activation for output layer
    cache = {'A1': A1, 'D1': D1, 'A2': A2, 'D2': D2, 'A3': A3, 'Z1': Z1, 'Z2': Z2, 'Z3': Z3}
    return A3, cache

# Function to compute cost for output of neural network
def compute_cost(A3, Y):
    m = Y.shape[0]
    clipped_probs = np.clip(A3, 1e-8, 1 - 1e-8)
    log_probs = -np.log(clipped_probs) * Y
    cost = np.sum(log_probs) / m
    return cost

# Function for backward propagation in neural network
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[0]
    W1, W2, W3 = parameters['W1'], parameters['W2'], parameters['W3']
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    D1, D2 = cache['D1'], cache['D2']
    Z1, Z2 = cache['Z1'], cache['Z2']

    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = np.dot(dZ3, W3.T)
    dA2 *= D2  # Apply dropout mask
    dZ2 = dA2 * (Z2 > 0)  # ReLU derivative
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dA1 *= D1  # Apply dropout mask
    dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2, 'dW3': dW3, 'db3': db3}
    return grads

# Function to update neural network parameters using the Adam optimizer
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    v_corrected = {}
    s_corrected = {}

    for param_name in parameters:
        # Update first moment estimate
        v["d" + param_name] = beta1 * v["d" + param_name] + (1 - beta1) * grads["d" + param_name]
        # Update second moment estimate
        s["d" + param_name] = beta2 * s["d" + param_name] + (1 - beta2) * (grads["d" + param_name] ** 2)

        # Correct bias in first moment
        v_corrected["d" + param_name] = v["d" + param_name] / (1 - beta1 ** t)
        # Correct bias in second moment
        s_corrected["d" + param_name] = s["d" + param_name] / (1 - beta2 ** t)

        # Update parameters
        parameters[param_name] -= learning_rate * v_corrected["d" + param_name] / (np.sqrt(s_corrected["d" + param_name]) + epsilon)

    return parameters, v, s

# Main function to run the neural network model for a specified dataset
def neural_network_model(X, Y, hidden_size1, hidden_size2, num_epochs, learning_rate, output_size, keep_prob=0.8):
    np.random.seed(1)  # Ensure reproducibility
    input_size = X.shape[1]
    parameters = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)
    v, s = {}, {}
    for param_name in parameters:
        v["d" + param_name] = np.zeros_like(parameters[param_name])
        s["d" + param_name] = np.zeros_like(parameters[param_name])

    Y_one_hot = convert_to_one_hot(Y, output_size)  # Convert labels to one-hot encoding

    for epoch in range(num_epochs):
        A3, cache = forward_propagation(X, parameters, keep_prob)
        cost = compute_cost(A3, Y_one_hot)  # Use one-hot encoded labels
        grads = backward_propagation(parameters, cache, X, Y_one_hot)  # Use one-hot encoded labels
        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, epoch + 1, learning_rate)
        if epoch % 100 == 0:
            print(f"Cost after iteration {epoch}: {cost:.6f}")
    return parameters

# Function to predict the output using the trained neural network
def predict(X, parameters):
    A3, cache = forward_propagation(X, parameters, keep_prob=1.0)  # No dropout during prediction
    predictions = np.argmax(A3, axis=1)
    return predictions

# Function to write predictions to a CSV file
def write_predictions(predictions, file_path='output.csv'):
    pd.DataFrame(predictions + 1, columns=['BEDS']).to_csv(file_path, index=False)

# Function to load and preprocess the training and testing data
def load_and_preprocess_data(train_data_path, test_data_path, train_label_path):
    train_data = load_csv(train_data_path)
    test_data = load_csv(test_data_path)
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    
    combined_data = combined_data.drop(columns=['ADDRESS', 'MAIN_ADDRESS', 'BROKERTITLE', 'STATE', 'LOCALITY', 'SUBLOCALITY', 'ADMINISTRATIVE_AREA_LEVEL_2', 'STREET_NAME', 'LONG_NAME', 'FORMATTED_ADDRESS'], errors='ignore')
    combined_data = one_hot_encode(combined_data)
    numerical_columns = ['PRICE', 'BATH', 'PROPERTYSQFT', 'LATITUDE', 'LONGITUDE']
    combined_data = normalize(combined_data, numerical_columns)
    preprocessed_train_data = combined_data.iloc[:len(train_data)]
    preprocessed_test_data = combined_data.iloc[len(train_data):]
    train_labels = load_csv(train_label_path).values
    train_labels = train_labels.astype(int) - 1  # Adjust labels to be zero-indexed
    return preprocessed_train_data.values, train_labels, preprocessed_test_data.values

# Main function to execute the entire workflow
def main(train_data_path, train_label_path, test_data_path, output_file_path):
    train_data, train_labels, test_data = load_and_preprocess_data(train_data_path, test_data_path, train_label_path)
    train_data = np.array(train_data, dtype=float)
    train_labels = train_labels.flatten()
    
    # Define hyperparameters
    hidden_size1 = 32
    hidden_size2 = 32
    num_epochs = 400
    learning_rate = 3e-3
    output_size = np.max(train_labels) + 1
    
    # Continue with training and testing
    test_data = np.array(test_data, dtype=float)
    trained_parameters = neural_network_model(train_data, train_labels, hidden_size1, hidden_size2, num_epochs, learning_rate, output_size)
    predictions = predict(test_data, trained_parameters)
    write_predictions(predictions, output_file_path)

# Running the script
if __name__ == "__main__":
    main('train_data1.csv', 'train_label1.csv', 'test_data1.csv', 'output1.csv')
    main('train_data2.csv', 'train_label2.csv', 'test_data2.csv', 'output2.csv')
    main('train_data3.csv', 'train_label3.csv', 'test_data3.csv', 'output3.csv')
    main('train_data4.csv', 'train_label4.csv', 'test_data4.csv', 'output4.csv')
    main('train_data5.csv', 'train_label5.csv', 'test_data5.csv', 'output5.csv')