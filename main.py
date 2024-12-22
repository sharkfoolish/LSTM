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


class CustomLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wc = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)

        self.bf = np.zeros((hidden_dim, 1))
        self.bi = np.zeros((hidden_dim, 1))
        self.bo = np.zeros((hidden_dim, 1))
        self.bc = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def forward_pass(self, xt, ht_1, ct_1):
        concat = np.vstack((ht_1, xt))
        f_t = sigmoid(np.dot(self.Wf, concat) + self.bf)
        i_t = sigmoid(np.dot(self.Wi, concat) + self.bi)
        cp_t = tanh(np.dot(self.Wc, concat) + self.bc)
        o_t = sigmoid(np.dot(self.Wo, concat) + self.bo)
        c_t = f_t * ct_1 + i_t * cp_t
        h_t = o_t * tanh(c_t)
        y_pred = np.dot(self.Wy, h_t) + self.by
        return h_t, c_t, y_pred, f_t, i_t, cp_t, o_t

    def backward_pass(self, xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t):
        dy = 2 * (y_pred - yt)
        dWy = np.dot(dy, h_t.T)
        dby = dy

        dh_t = np.dot(self.Wy.T, dy)

        do_t = dh_t * tanh(c_t) * sigmoid_derivative(np.dot(self.Wo, np.vstack((ht_1, xt))) + self.bo)
        dc_t = dh_t * o_t * tanh_derivative(tanh(c_t))
        df_t = dc_t * ct_1 * sigmoid_derivative(np.dot(self.Wf, np.vstack((ht_1, xt))) + self.bf)
        di_t = dc_t * cp_t * sigmoid_derivative(np.dot(self.Wi, np.vstack((ht_1, xt))) + self.bi)
        dcp_t = dc_t * i_t * tanh_derivative(np.dot(self.Wc, np.vstack((ht_1, xt))) + self.bc)

        dWf = np.dot(df_t, np.vstack((ht_1, xt)).T)
        dWi = np.dot(di_t, np.vstack((ht_1, xt)).T)
        dWo = np.dot(do_t, np.vstack((ht_1, xt)).T)
        dWc = np.dot(dcp_t, np.vstack((ht_1, xt)).T)
        dbf = df_t
        dbi = di_t
        dbo = do_t
        dbc = dcp_t

        self.Wf -= self.learning_rate * dWf
        self.Wi -= self.learning_rate * dWi
        self.Wo -= self.learning_rate * dWo
        self.Wc -= self.learning_rate * dWc
        self.Wy -= self.learning_rate * dWy
        self.bf -= self.learning_rate * dbf
        self.bi -= self.learning_rate * dbi
        self.bo -= self.learning_rate * dbo
        self.bc -= self.learning_rate * dbc
        self.by -= self.learning_rate * dby

    def train(self, X, y, epochs):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            ht_1 = np.zeros((self.hidden_dim, 1))
            ct_1 = np.zeros((self.hidden_dim, 1))

            for i in range(len(X)):
                xt = X[i]
                yt = y[i]

                h_t, c_t, y_pred, f_t, i_t, cp_t, o_t = self.forward_pass(xt, ht_1, ct_1)
                loss = (y_pred - yt) ** 2
                total_loss += loss
                self.backward_pass(xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t)

                ht_1 = h_t
                ct_1 = c_t

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

lstm_model = CustomLSTM(input_dim=data.shape[1], hidden_dim=50, output_dim=1)
lstm_model.train(data, target, epochs=5000)
predictions = lstm_model.predict(data)
mse_of_predictions = mse(target, predictions)
mae_of_predictions = mae(target, predictions)
r2_of_predictions = r2(target, predictions)

print(f"CustomLSTM Model - MSE: {mse_of_predictions:.4f}, MAE: {mae_of_predictions:.4f}, R2: {r2_of_predictions:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(target, label='Actual Values', marker='o')
plt.plot(predictions, label='Predicted Values Of CustomLSTM', marker='x')
plt.xlabel('Sample')
plt.ylabel('House Value')
plt.title('Actual vs. Predicted Values Of CustomLSTM')
plt.show()
