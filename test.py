import pandas as pd
import torch
from models.Transformer import Model
from configs.three_hour.Transformer_Configs import Configs

file_path = './data_input.csv'
data = pd.read_csv(file_path, parse_dates=['date'])
input_series = data['soilMoisture'].values
input_tensor = torch.tensor(input_series, dtype=torch.float32)
input_tensor = input_tensor.view(1, -1, 1)
print(f"Input tensor shape: {input_tensor.shape}")
configs = Configs()
model = Model(configs)
model.load_state_dict(torch.load('final_model_3hour.pth', map_location=torch.device('cpu'), weights_only=True))
model.to(torch.device('cpu'))
with torch.no_grad():
    input_tensor = input_tensor.to(torch.device('cpu'))
    output = model(input_tensor)
print(output)