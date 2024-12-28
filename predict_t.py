import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from data_provider.data_factory import data_provider
from models.Transformer import Model
from configs.three_hour.Transformer_Configs import Configs
from exp.exp_main import Exp_Main
from utils.tools import dotdict

class PredictionTool:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path


        # Định nghĩa tham số
        self.args = dotdict()
        self.args.target = 'soilMoisture'
        self.args.des = 'test'
        self.args.dropout = 0.05
        self.args.data_path = data_path
        self.args.num_workers = 10
        self.args.gpu = 0
        self.args.lradj = 'type1'
        self.args.devices = '0'
        self.args.use_gpu = False
        self.args.use_multi_gpu = False


        self.args.freq = 'h'
        self.args.checkpoints = './checkpoints/'
        self.args.bucket_size = 4
        self.args.n_hashes = 4
        self.args.seq_len = 36
        self.args.label_len = 36
        self.args.pred_len = 1
        self.args.e_layers = 2
        self.args.d_layers = 1
        self.args.n_heads = 8
        self.args.factor = 1
        self.args.d_model = 512
        self.args.des = 'Exp'
        self.args.itr = 1
        self.args.d_ff = 2048
        self.args.moving_avg = 25
        self.args.distil = True
        self.args.output_attention = False
        self.args.patience= 3
        self.args.learning_rate = 0.0001
        self.args.batch_size = 32
        self.args.embed = 'timeF'
        self.args.activation = 'gelu'
        self.args.use_amp = False
        self.args.loss = 'mse'

        self.args.train_epochs = 4
        self.args.is_training = True
        self.args.enc_in = 2
        self.args.dec_in = 2
        self.args.c_out = 2
        self.args.target = "soilMoisture"
        self.args.root_path = './'
        self.args.data_path =data_path
        self.args.model_id='temperature_humidity'
        self.args.model = 'Transformer'
        #args.model = 'Informer'
        self.args.data = 'custom'
        self.args.features = 'MS'

        # Khởi tạo mô hình
        self.exp = Exp_Main(self.args)
        self.configs = Configs()
        self.exp.model = Model(self.configs)
        self.exp.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        self.exp.model.to(self.args.device)
        self.exp.model.eval()
    
    def inverse_transform(self,x):
        file_path = 'dataset/temperature_humidity_tran1.csv'

        data = pd.read_csv(file_path)
        columns_to_scale = ['soilMoisture']
        data_to_scale = data[columns_to_scale]

        scaler = StandardScaler()
        scaler.fit(data_to_scale)
        x = np.squeeze(x)  # Loại bỏ các chiều có kích thước 1 (nếu có)
        x = x.reshape(-1, 1)  # Định hình lại mảng để thành dạng 2 chiều phù hợp với inverse_transform
        x_original = scaler.inverse_transform(x)
        return x_original

    def predict(self):
        """Dự đoán và trả về kết quả."""
        # Lấy dữ liệu dự đoán
        pred_data, pred_loader = self.exp._get_data(flag='pred')

        # Dự đoán
        preds = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)
                outputs, batch_y = self.exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        preds_original= self.inverse_transform(preds)
        print(preds)
        print(self.inverse_transform(preds))

        return preds_original
