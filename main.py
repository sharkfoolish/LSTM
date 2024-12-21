import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


class CustomLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        self.Wf = np.random.uniform(-0.1, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.Wi = np.random.uniform(-0.1, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.Wo = np.random.uniform(-0.1, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.Wc = np.random.uniform(-0.1, 0.1, (hidden_dim, input_dim + hidden_dim))
        self.Wy = np.random.uniform(-0.1, 0.1, (output_dim, hidden_dim))

    def forward_pass(self, xt, ht_1, ct_1):
        concat = np.vstack((ht_1, xt))
        f_t = sigmoid(np.dot(self.Wf, concat))
        i_t = sigmoid(np.dot(self.Wi, concat))
        cp_t = tanh(np.dot(self.Wc, concat))
        o_t = sigmoid(np.dot(self.Wo, concat))
        c_t = f_t * ct_1 + i_t * cp_t
        h_t = o_t * tanh(c_t)
        y_pred = np.dot(self.Wy, h_t)
        return h_t, c_t, y_pred, f_t, i_t, cp_t, o_t

    def backward_pass(self, xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t):
        dy = 2 * (y_pred - yt)
        dWy = np.dot(dy, h_t.T)
        dby = dy

        dh_t = np.dot(self.Wy.T, dy)

        do_t = dh_t * tanh(c_t) * sigmoid_derivative(np.dot(self.Wo, np.vstack((ht_1, xt))))
        dc_t = dh_t * o_t * tanh_derivative(tanh(c_t))
        df_t = dc_t * ct_1 * sigmoid_derivative(np.dot(self.Wf, np.vstack((ht_1, xt))))
        di_t = dc_t * cp_t * sigmoid_derivative(np.dot(self.Wi, np.vstack((ht_1, xt))))
        dcp_t = dc_t * i_t * tanh_derivative(np.dot(self.Wc, np.vstack((ht_1, xt))))

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
