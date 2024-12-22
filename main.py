import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def train_test_split(data, target, train_size, test_size, random_state=None):

    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError("The sum of train_size and test_size must be 1.0")

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(data))
    train_end = int(len(data) * train_size)

    data_train, data_test = data[indices[:train_end]], data[indices[train_end:]]
    target_train, target_test = target[indices[:train_end]], target[indices[train_end:]]

    return data_train, data_test, target_train, target_test


# Xavier 初始化，用於初始化權重
def xavier_init(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))


# 定義自製 LSTM 類別
class CustomLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, reg_lambda=0.01, decay_factor=0.01):
        self.losses = []
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.reg_lambda = reg_lambda
        self.initialize_weights()
        self.initialize_biases()

    def initialize_weights(self):
        self.weights = {
            'forget_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'input_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'output_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'cell_state': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'output_layer': xavier_init(self.output_dim, self.hidden_dim)
        }

    def initialize_biases(self):
        self.biases = {
            'forget_gate': np.zeros((self.hidden_dim, 1)),
            'input_gate': np.zeros((self.hidden_dim, 1)),
            'output_gate': np.zeros((self.hidden_dim, 1)),
            'cell_state': np.zeros((self.hidden_dim, 1)),
            'output_layer': np.zeros((self.output_dim, 1))
        }

    def learning_rate_decay(self, epoch):
        self.learning_rate = self.init_learning_rate / (1 + self.decay_factor * epoch)

    def forward_pass(self, xt, ht_1, ct_1):
        concat = np.vstack((ht_1, xt))
        f_t = sigmoid(np.dot(self.weights['forget_gate'], concat) + self.biases['forget_gate'])
        i_t = sigmoid(np.dot(self.weights['input_gate'], concat) + self.biases['input_gate'])
        cp_t = tanh(np.dot(self.weights['cell_state'], concat) + self.biases['cell_state'])
        o_t = sigmoid(np.dot(self.weights['output_gate'], concat) + self.biases['output_gate'])
        c_t = f_t * ct_1 + i_t * cp_t
        h_t = o_t * tanh(c_t)
        y_pred = np.dot(self.weights['output_layer'], h_t) + self.biases['output_layer']
        return h_t, c_t, y_pred, f_t, i_t, cp_t, o_t

    def backward_pass(self, xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t):
        concat = np.vstack((ht_1, xt))
        dy = 2 * (y_pred - yt)
        dWy = np.dot(dy, h_t.T)
        dby = dy
        dh_t = np.dot(self.weights['output_layer'].T, dy)

        do_t = dh_t * tanh(c_t) * sigmoid_derivative(np.dot(self.weights['output_gate'], concat) + self.biases['output_gate'])
        dc_t = dh_t * o_t * tanh_derivative(tanh(c_t))
        df_t = dc_t * ct_1 * sigmoid_derivative(np.dot(self.weights['forget_gate'], concat) + self.biases['forget_gate'])
        di_t = dc_t * cp_t * sigmoid_derivative(np.dot(self.weights['input_gate'], concat) + self.biases['input_gate'])
        dcp_t = dc_t * i_t * tanh_derivative(np.dot(self.weights['cell_state'], concat) + self.biases['cell_state'])

        dWf = np.dot(df_t, concat.T)
        dWi = np.dot(di_t, concat.T)
        dWo = np.dot(do_t, concat.T)
        dWc = np.dot(dcp_t, concat.T)
        dbf = df_t
        dbi = di_t
        dbo = do_t
        dbc = dcp_t

        self.weights['forget_gate'] -= self.learning_rate * (dWf + self.reg_lambda * self.weights['forget_gate'] * 2)
        self.weights['input_gate'] -= self.learning_rate * (dWi + self.reg_lambda * self.weights['input_gate'] * 2)
        self.weights['output_gate'] -= self.learning_rate * (dWo + self.reg_lambda * self.weights['output_gate'] * 2)
        self.weights['cell_state'] -= self.learning_rate * (dWc + self.reg_lambda * self.weights['output_gate'] * 2)
        self.weights['output_layer'] -= self.learning_rate * (dWc + self.reg_lambda * self.weights['output_gate'] * 2)
        self.biases['forget_gate'] -= self.learning_rate * dbf
        self.biases['input_gate'] -= self.learning_rate * dbi
        self.biases['output_gate'] -= self.learning_rate * dbo
        self.biases['cell_state'] -= self.learning_rate * dbc
        self.biases['output_layer'] -= self.learning_rate * dby

    def train(self, data_train, target_train, epochs):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            ht_1 = np.zeros((self.hidden_dim, 1))
            ct_1 = np.zeros((self.hidden_dim, 1))

            for i in range(len(data_train)):
                xt = data_train[i]
                yt = target_train[i]
                h_t, c_t, y_pred, f_t, i_t, cp_t, o_t = self.forward_pass(xt, ht_1, ct_1)
                loss = (y_pred - yt) ** 2
                total_loss += loss
                self.backward_pass(xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t)

                ht_1 = h_t
                ct_1 = c_t

            avg_loss = total_loss / len(data_train)
            self.losses.append(avg_loss)
            self.learning_rate_decay(epoch)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        ht_1 = np.zeros((self.hidden_dim, 1))
        ct_1 = np.zeros((self.hidden_dim, 1))
        predictions = []

        for xt in X:
            h_t, c_t, y_pred, _, _, _, _ = self.forward_pass(xt, ht_1, ct_1)
            predictions.append(y_pred)
            ht_1 = h_t
            ct_1 = c_t

        return np.array(predictions).flatten()


california_housing_data_set = fetch_california_housing()
data = california_housing_data_set.data[:1000]
target = california_housing_data_set.target[:1000]

scaler = StandardScaler()
data = scaler.fit_transform(data)

data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.6, test_size=0.2, random_state=1000)

lstm_model = CustomLSTM(input_dim=data_train.shape[1], hidden_dim=50, output_dim=1)
lstm_model.train(data_train, target_train, epochs=5000)
predictions = lstm_model.predict(data_test)
mse_of_predictions = mse(target_test, predictions)
mae_of_predictions = mae(target_test, predictions)
r2_of_predictions = r2(target_test, predictions)

print(f"CustomLSTM Model - MSE: {mse_of_predictions:.4f}, MAE: {mae_of_predictions:.4f}, R2: {r2_of_predictions:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(target_test, label='Actual Values', marker='o')
plt.plot(predictions, label='Predicted Values Of CustomLSTM', marker='x')
plt.xlabel('Sample')
plt.ylabel('House Value')
plt.title('Actual vs. Predicted Values Of CustomLSTM')
plt.show()
