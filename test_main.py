import unittest
import numpy as np
from main import CustomLSTM, mse

class TestCustomLSTM(unittest.TestCase):

    def setUp(self):
        # 初始化測試資料和模型
        self.input_dim = 10
        self.hidden_dim = 5
        self.output_dim = 1
        self.model = CustomLSTM(self.input_dim, self.hidden_dim, self.output_dim)

        # 測試用的輸入與目標輸出
        self.X_sample = np.random.randn(10, self.input_dim)  # 10 個時間步，每步有 input_dim 個特徵
        self.y_sample = np.random.randn(10, self.output_dim)  # 對應的目標值

    def test_initialization(self):
        # 測試模型參數是否正確初始化
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)

    def test_forward_pass(self):
        # 測試前向傳播是否能正常運作
        ht_1 = np.zeros((self.hidden_dim, 1))
        ct_1 = np.zeros((self.hidden_dim, 1))

        for t in range(self.X_sample.shape[0]):
            xt = self.X_sample[t].reshape(-1, 1)
            h_t, c_t, y_pred, f_t, i_t, c_tilde, o_t = self.model.forward_pass(xt, ht_1, ct_1)

            # 確認輸出的維度是否正確
            self.assertEqual(h_t.shape, (self.hidden_dim, 1))
            self.assertEqual(c_t.shape, (self.hidden_dim, 1))
            self.assertEqual(y_pred.shape, (self.output_dim, 1))

    def test_training_step(self):
        # 測試單步訓練是否能正常運作
        ht_1 = np.zeros((self.hidden_dim, 1))
        ct_1 = np.zeros((self.hidden_dim, 1))
        learning_rate = 0.01
    
        for t in range(self.X_sample.shape[0]):
            xt = self.X_sample[t].reshape(-1, 1)
            yt = self.y_sample[t].reshape(-1, 1)
            h_t, c_t, y_pred, f_t, i_t, cp_t, o_t = self.model.forward_pass(xt, ht_1, ct_1)
    
            # 計算損失
            loss = mse(y_pred, yt)
    
            # 反向傳播
            dWy, dby, do_t, dc_t, df_t, di_t, dcp_t = self.model.backward_pass(xt, ht_1, ct_1, h_t, c_t, y_pred, yt, f_t, i_t, cp_t, o_t)
    
            # 檢查損失是否為非負值
            self.assertGreaterEqual(loss, 0)
    
            # 檢查反向傳播的梯度是否是合理的 (非空)
            self.assertTrue(dWy is not None)
            self.assertTrue(dby is not None)
            self.assertTrue(do_t is not None)
            self.assertTrue(dc_t is not None)
            self.assertTrue(df_t is not None)
            self.assertTrue(di_t is not None)
            self.assertTrue(dcp_t is not None)


    def test_evaluate(self):
        # 測試評估方法是否能正確運作
        avg_loss = self.model.evaluate(self.X_sample, self.y_sample)

        # 檢查損失是否為非負值
        self.assertGreaterEqual(avg_loss, 0)

if __name__ == "__main__":
    unittest.main()
