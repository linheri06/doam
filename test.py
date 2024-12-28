import pandas as pd
import torch
from models.Transformer import Model
from configs.three_hour.Transformer_Configs import Configs

file_path = './data_input.csv'
data = pd.read_csv(file_path, parse_dates=['date'])

input_series = data['soilMoisture'].values
#time_features = data[['date']].apply(lambda x: [x.dt.minute,x.dt.hour, x.dt.day, x.dt.month], axis=1).values
#print(time_features.size)


# Trích xuất các đặc trưng thời gian
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month

# Tạo tensor đặc trưng thời gian (bao gồm giờ, phút, ngày, tháng)
time_features = data[['hour', 'minute', 'day', 'month']].values  # Shape: (num_samples, 4)
time_tensor = torch.tensor(time_features, dtype=torch.float32).view(1, -1, 4)  # Shape: (1, seq_len, 4)


input_tensor = torch.tensor(input_series, dtype=torch.float32)
input_tensor = input_tensor.view(1, -1, 1)
#time_tensor = torch.tensor(time_features, dtype=torch.float32).view(1, -1, 3)


print(f"Input tensor shape: {input_tensor.shape}")
# configs = Configs()
# model = Model(configs)
# model.load_state_dict(torch.load('final_model_3hour.pth', map_location=torch.device('cpu'), weights_only=True))
# model.to(torch.device('cpu'))

# # 4. Prepare inputs for the model
# x_enc = input_tensor  # Encoder input
# x_mark_enc = time_tensor  # Time features for encoder
# x_dec = input_tensor[:, -configs.pred_len:, :]  # Decoder input (e.g., last `label_len` timesteps)
# x_mark_dec = time_tensor[:, -configs.pred_len:, :]  # Time features for decoder


# with torch.no_grad():
#     # input_tensor = input_tensor.to(torch.device('cpu'))
#     # output = model(input_tensor)
#     x_enc = x_enc.to(torch.device('cpu'))
#     x_mark_enc = x_mark_enc.to(torch.device('cpu'))
#     x_dec = x_dec.to(torch.device('cpu'))
#     x_mark_dec = x_mark_dec.to(torch.device('cpu'))

#     output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
# print(output)
configs = Configs()
model = Model(configs)
model.load_state_dict(torch.load('final_model_3hour.pth', map_location=torch.device('cpu'), weights_only=True))
model.to(torch.device('cpu'))
with torch.no_grad():
    input_tensor = input_tensor.to(torch.device('cpu'))
    output = model(input_tensor)
print(output)