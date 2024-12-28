from data_provider.data_factory import data_provider
import pandas as pd
import torch
from models.Transformer import Model
from configs.three_hour.Transformer_Configs import Configs

from exp.exp_main import Exp_Main
from utils.tools import dotdict
import numpy as np

def main():
    args = dotdict()
    args.target = 'OT'
    args.des = 'test'
    args.dropout = 0.05
    args.num_workers = 10
    args.gpu = 0
    args.lradj = 'type1'
    args.devices = '0'
    args.use_gpu = False
    args.use_multi_gpu = False
    # if args.use_gpu and args.use_multi_gpu: #是否使用多卡的判断
    #     args.dvices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    args.bucket_size = 4
    args.n_hashes = 4
    args.seq_len = 36
    args.label_len = 36
    args.pred_len = 1
    args.e_layers = 2
    args.d_layers = 1
    args.n_heads = 8
    args.factor = 1
    args.d_model = 512
    args.des = 'Exp'
    args.itr = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.distil = True
    args.output_attention = False
    args.patience= 3
    args.learning_rate = 0.0001
    args.batch_size = 32
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.use_amp = False
    args.loss = 'mse'

    args.train_epochs = 4
    args.is_training = True
    args.enc_in = 2
    args.dec_in = 2
    args.c_out = 2
    args.target = "soilMoisture"
    args.root_path = './dataset/'
    args.data_path ='data_input.csv'
    args.model_id='temperature_humidity'
    args.model = 'Transformer'
    #args.model = 'Informer'
    args.data = 'custom'
    args.features = 'MS'

    Exp = Exp_Main

    exp = Exp(args)

    configs = Configs()
    exp.model = Model(configs)
    exp.model.load_state_dict(torch.load('final_model.pth', map_location=torch.device('cpu'), weights_only=True))
    exp.model.to(torch.device('cpu'))
    exp.model.eval()

    pred_data, pred_loader = exp._get_data(flag='pred')
    # preds = []
    # #exp.predict()

    # print(pred_data.data_x)

    preds = []
    print(enumerate(pred_loader))

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)
            print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            outputs, batch_y = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)
    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    print(preds)
    print(invension(preds))


def invension(x):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

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


if __name__ == "__main__":
    main()