import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# Sigmoid 函數：常用於神經網絡中，將輸入值壓縮至 0 到 1 之間
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Tanh 函數：雙曲正切函數，將輸入值壓縮至 -1 到 1 之間
def tanh(x):
    return np.tanh(x)


# Sigmoid 函數的導數
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Tanh 函數的導數
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# 均方誤差 (MSE) 損失函數
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 平均絕對誤差 (MAE) 損失函數
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# R2 決定係數，用於衡量回歸模型的預測效果
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# 資料集分割函數，將資料分成訓練集、驗證集和測試集
def train_val_test_split(data, target, train_size, val_size, test_size, random_state=None):

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("The sum of train_size, val_size, and test_size must be 1.0")

    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(data))  # 隨機打亂資料索引
    train_end = int(len(data) * train_size)  # 訓練集結束索引
    val_end = train_end + int(len(data) * val_size)  # 驗證集結束索引

    data_train, data_val, data_test = data[indices[:train_end]], data[indices[train_end:val_end]], data[indices[val_end:]]
    target_train, target_val, target_test = target[indices[:train_end]], target[indices[train_end:val_end]], target[indices[val_end:]]

    return data_train, data_val, data_test, target_train, target_val, target_test


# Xavier 初始化，用於初始化權重
def xavier_init(rows, cols):
    return np.random.randn(rows, cols) * np.sqrt(2.0 / (rows + cols))


# 定義自製 LSTM 類別
class CustomLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, reg_lambda=0.01, decay_factor=0.01, patience=10):
        self.losses = []  # 訓練過程中的損失
        self.val_losses = []  # 驗證過程中的損失
        self.input_dim = input_dim  # 輸入維度
        self.hidden_dim = hidden_dim  # 隱藏層維度
        self.output_dim = output_dim  # 輸出維度
        self.init_learning_rate = learning_rate  # 初始學習率
        self.learning_rate = learning_rate  # 當前學習率
        self.decay_factor = decay_factor  # 學習率衰減參數
        self.reg_lambda = reg_lambda  # 正則化參數
        self.patience = patience  # 早停的耐心次數
        self.best_val_loss = float('inf')  # 最佳驗證損失
        self.patience_counter = 0  # 早停計數器
        self.initialize_weights()  # 初始化權重
        self.initialize_biases()  # 初始化偏差

    # 初始化權重
    def initialize_weights(self):
        self.weights = {
            'forget_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'input_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'output_gate': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'cell_state': xavier_init(self.hidden_dim, self.input_dim + self.hidden_dim),
            'output_layer': xavier_init(self.output_dim, self.hidden_dim)
        }

    # 初始化Biases
    def initialize_biases(self):
        self.biases = {
            'forget_gate': np.zeros((self.hidden_dim, 1)),
            'input_gate': np.zeros((self.hidden_dim, 1)),
            'output_gate': np.zeros((self.hidden_dim, 1)),
            'cell_state': np.zeros((self.hidden_dim, 1)),
            'output_layer': np.zeros((self.output_dim, 1))
        }

    # 學習率衰減
    def learning_rate_decay(self, epoch):
        self.learning_rate = self.init_learning_rate / (1 + self.decay_factor * epoch)

    def forward_pass(self, xt, ht_1, ct_1):
        concat = np.vstack((ht_1, xt))  # 拼接前一時刻的隱藏狀態和當前輸入
        f_t = sigmoid(np.dot(self.weights['forget_gate'], concat) + self.biases['forget_gate'])  # 忘記門
        i_t = sigmoid(np.dot(self.weights['input_gate'], concat) + self.biases['input_gate'])  # 輸入門
        cp_t = tanh(np.dot(self.weights['cell_state'], concat) + self.biases['cell_state'])  # 候選記憶單元
        o_t = sigmoid(np.dot(self.weights['output_gate'], concat) + self.biases['output_gate'])  # 輸出門
        c_t = f_t * ct_1 + i_t * cp_t  # 記憶單元狀態更新
        h_t = o_t * tanh(c_t)  # 隱藏狀態更新
        y_pred = np.dot(self.weights['output_layer'], h_t) + self.biases['output_layer']  # 輸出層預測
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

    # 訓練模型
    def train(self, data_train, target_train, epochs):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            ht_1 = np.zeros((self.hidden_dim, 1))
            ct_1 = np.zeros((self.hidden_dim, 1))

            for i in range(len(data_train)):
                xt = data_train[i]
                yt = target_train[i]
                h_t, c_t, y_pred, f_t, i_t, cp_t, o_t = self.forward_pass(xt, ht_1, ct_1)
                loss = mse(y_pred, yt)
                total_loss += loss
                self.backward_pass(xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t)

                ht_1 = h_t
                ct_1 = c_t

            avg_loss = total_loss / len(data_train)
            self.losses.append(avg_loss)
            self.learning_rate_decay(epoch)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

            val_loss = self.evaluate(data_val, target_val)
            self.val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")

            # 早停機制
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = copy.deepcopy(self.__dict__)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.__dict__.update(self.best_weights)
                    break

    def evaluate(self, X, y):
        total_loss = 0
        ht_1 = np.zeros((self.hidden_dim, 1))
        ct_1 = np.zeros((self.hidden_dim, 1))

        for i in range(len(X)):
            xt = X[i].reshape(-1, 1)
            yt = y[i].reshape(-1, 1)
            h_t, c_t, y_pred, _, _, _, _ = self.forward_pass(xt, ht_1, ct_1)

            loss = mse(y_pred, yt)
            total_loss += loss

            ht_1 = h_t
            ct_1 = c_t

        avg_loss = total_loss / len(X)
        return avg_loss

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


# 獲得 California housing 資料集
california_housing_data_set = fetch_california_housing()
data = california_housing_data_set.data[:1000]
target = california_housing_data_set.target[:1000]

# 資料標準化
scaler = StandardScaler()
data = scaler.fit_transform(data)

# 訓練、驗證、測試資料分割
data_train, data_val, data_test, target_train, target_val, target_test = train_val_test_split(data, target, train_size=0.6, val_size=0.2, test_size=0.2, random_state=1000)

# 訓練 LSTM 模型
lstm_model = CustomLSTM(input_dim=data_train.shape[1], hidden_dim=50, output_dim=1)
lstm_model.train(data_train, target_train, epochs=5000)
predictions = lstm_model.predict(data_test)
print(f"CustomLSTM Model - MSE: {mse(target_test, predictions):.4f}, MAE: {mae(target_test, predictions):.4f}, R2: {r2(target_test, predictions):.4f}")

# 比對實際值和預測值
plt.figure(figsize=(12, 6))
plt.plot(target_test, label='Actual Values', marker='o')
plt.plot(predictions, label='Predicted Values Of CustomLSTM', marker='x')
plt.xlabel('Sample')
plt.ylabel('House Value')
plt.title('Actual vs. Predicted Values Of CustomLSTM')
plt.show()
