from torch_geometric.data import DataLoader
from GCN2 import PatientGraphClassifier, PatientResidualGCN, GATNet
import torch
import pandas as pd
import os
import numpy as np
from Create_graph2 import create_graph, create_graph_with_statistical_threshold
from torch_geometric.data import DataLoader
from lifelines.utils import concordance_index
from loss_metric import NegativeLogLikelihood, AFTLoss, DeepSurvLoss


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

##########################################
local_path = './'
model_file = './model.pth'


valid_info = pd.read_excel('./data/patient_info.xlsx')
valid_labels = {row['Patient_num']: {'Status': row['Status'], 'OS':row['OS']} for index, row in valid_info.iterrows()}
print(valid_labels)

valid_ids = valid_info['Patient_num'].tolist()
print(valid_ids)
train_list = []
valid_list = []
pt_path = './data'

def test(model, device, loader):
    model.eval()
    all_risk = []
    all_os = []
    all_status = []
    predict_risk =[]
    epoch_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            print('out:',out.squeeze().tolist())
            status_list = data.y[::2]
            os_list = data.y[1::2]

            if np.isnan(out.cpu().tolist()[0][0]):
                continue
            all_risk.append(out.detach().cpu().numpy())
            all_os.append(os_list.cpu().numpy())
            all_status.append(status_list.cpu().numpy())
            predict_risk.append(out.squeeze().tolist())

        all_risk = np.concatenate(all_risk)
        all_os = np.concatenate(all_os)
        all_status = np.concatenate(all_status)

        ci = concordance_index(all_os, -all_risk, all_status)

    return ci, all_os, all_status, predict_risk

for pid in valid_ids:
    print(pid)
    #########
    # #######加载已经生成的图结构，#############
    file_name = f'{pid}.pt'
    file_path = os.path.join(pt_path, file_name)
    print(file_path)
    valid_graph_data = torch.load(file_path)

    valid_list.append(valid_graph_data)
print('测试图制作完成！')

val_loader = DataLoader(valid_list, batch_size=1, shuffle=False, drop_last=False)

model = GATNet(num_node_features=1074)

train_CI = []
valid_CI = []

model.to(device)
model.load_state_dict(torch.load(model_file))  # 加载训练好的模型参数

valid_ci, os, status, risk = test(model, device, val_loader)
print('os:', os)
print('status:', status)
print('risk:',risk)
result = pd.DataFrame({'Patient_num':valid_ids, 'OS':os, 'Status':status, 'Predicted':risk})

result.to_excel('patient_result.xlsx', index=False)

print(f"Validation CI: {valid_ci:.4f}")



