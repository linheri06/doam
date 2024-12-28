# import pandas as pd
# import torch
# from flask import Flask, jsonify, request, render_template
# from models.Transformer import Model
# from configs.three_hour.Transformer_Configs import Configs
# import numpy as np
# from exp.exp_main import Exp_Main
# from data_provider.data_factory import data_provider
# from utils.tools import dotdict

# app = Flask(__name__)

# configs = Configs()
# model_3h = Model(configs)
# model_3h.load_state_dict(torch.load('final_model_3hour.pth', map_location=torch.device('cpu')))
# model_3h.to(torch.device('cpu'))
# model_3h.eval()


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     predictions_1h = None
#     predictions_3h = None

#     if request.method == 'POST':
#         try:
#             input_series = [float(request.form[f'hour_{i}']) for i in range(1, 36)]
#         except ValueError:
#             return "Vui lòng nhập dữ liệu kiểu số."

#         Exp = Exp_Main

#         exp = Exp(Configs)
#         input_tensor = torch.tensor(input_series, dtype=torch.float32)
#         pred_data, pred_loader = exp._get_data(flag='pred')

#         preds = []
#         print(enumerate(pred_loader))
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
#                 batch_x = batch_x.float().to(exp.device)
#                 batch_y = batch_y.float()
#                 batch_x_mark = batch_x_mark.float().to(exp.device)
#                 batch_y_mark = batch_y_mark.float().to(exp.device)
#                 print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
#                 outputs, batch_y = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
#                 pred = outputs.detach().cpu().numpy()  # .squeeze()
#                 preds.append(pred)
#         preds = np.array(preds)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         input_tensor = input_tensor.view(1, -1, 1)

#         with torch.no_grad():
#             input_tensor = input_tensor.to(torch.device('cpu'))
            
#             output_3h = model_3h(input_tensor)
#             predictions_3h = output_3h.tolist()[0]

#     return render_template('index.html', predictions_1h=predictions_1h, predictions_3h=predictions_3h)

# if __name__ == '__main__':
#     app.run(debug=True)


import pandas as pd
import torch
from flask import Flask, jsonify, request, render_template
from models.Transformer import Model
from configs.three_hour.Transformer_Configs import Configs
import numpy as np
from exp.exp_main import Exp_Main
from data_provider.data_factory import data_provider
from utils.tools import dotdict

app = Flask(__name__)

# Khởi tạo model 3-hour
configs = Configs()
model_3h = Model(configs)
model_3h.load_state_dict(torch.load('final_model_3hour.pth', map_location=torch.device('cpu')))
model_3h.to(torch.device('cpu'))
model_3h.eval()


@app.route('/', methods=['GET', 'POST'])
def index():
    predictions_1h = None
    predictions_3h = None

    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form và chuyển thành mảng float
            input_series = [float(request.form[f'hour_{i}']) for i in range(1, 36)]
        except ValueError:
            return "Vui lòng nhập dữ liệu kiểu số."

        try:
            # Khởi tạo class Exp_Main
            exp = Exp_Main(configs)
            input_tensor = torch.tensor(input_series, dtype=torch.float32).view(1, -1, 1)

            # Lấy dữ liệu từ data_provider
            pred_data, pred_loader = exp._get_data(flag='pred')

            preds = []
            print("Đang xử lý dự đoán...")
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                    batch_x = batch_x.float().to(exp.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(exp.device)
                    batch_y_mark = batch_y_mark.float().to(exp.device)

                    outputs, _ = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    pred = outputs.detach().cpu().numpy()
                    preds.append(pred)

            # Gộp dự đoán
            preds = np.concatenate(preds, axis=0)

            # Dự đoán bằng model 3-hour
            with torch.no_grad():
                input_tensor = input_tensor.to(torch.device('cpu'))
                output_3h = model_3h(input_tensor)
                predictions_3h = output_3h.squeeze().tolist()

        except Exception as e:
            return f"Có lỗi xảy ra trong quá trình dự đoán: {str(e)}"

    return render_template('index.html', predictions_3h=predictions_3h)


if __name__ == '__main__':
    app.run(debug=True)
