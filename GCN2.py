import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv

class PatientGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes=1):
        super(PatientGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 512)
        self.conv4 = GCNConv(512, 1024)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))

        # 全局池化层
        x = global_mean_pool(x, batch)

        # 全连接层
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc1(x)  # 输出层
        x = self.fc2(x)  # 输出层


        return x

class PatientResidualGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes=1):
        super(PatientResidualGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.adjust_dim1 = torch.nn.Linear(num_node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.adjust_dim2 = torch.nn.Linear(128, 256)
        self.conv3 = GCNConv(256, 512)
        self.adjust_dim3 = torch.nn.Linear(256, 512)
        self.conv4 = GCNConv(512, 1024)
        self.adjust_dim4 = torch.nn.Linear(512, 1024)
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        identity = self.adjust_dim1(x)
        x = F.relu(self.conv1(x, edge_index)) + identity

        identity = self.adjust_dim2(x)
        x = F.relu(self.conv2(x, edge_index)) + identity

        identity = self.adjust_dim3(x)
        x = F.relu(self.conv3(x, edge_index)) + identity  # 新增的层

        identity = self.adjust_dim4(x)
        x = F.relu(self.conv4(x, edge_index)) + identity  # 新增的层

        x = global_mean_pool(x, data.batch)

        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc1(x)  # 输出层
        x = self.fc2(x)

        return x



class GATNet(torch.nn.Module):
    def __init__(self, num_node_features, num_classes=1, hidden_dim=8, dropout=0.0):
        super(GATNet, self).__init__()
        self.heads = 4
        # self.output_heads = 4
        self.droput = dropout

        self.gat_conv1 = GATConv(in_channels=num_node_features, out_channels=hidden_dim, heads=self.heads, concat=True,
                                 dropout=dropout)
        self.gat_conv2 = GATConv(in_channels=hidden_dim * self.heads, out_channels=hidden_dim, heads=self.heads, concat=True,
                                 dropout=dropout)
        self.gat_conv3 = GATConv(in_channels=hidden_dim * self.heads, out_channels=hidden_dim, heads=self.heads, concat=True,
                                 dropout=dropout)
        self.gat_conv4 = GATConv(in_channels=hidden_dim * self.heads, out_channels=hidden_dim, heads=self.heads, concat=True,
                                 dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim * self.heads, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat_conv1(x, edge_index))
        x = F.elu(self.gat_conv2(x, edge_index))
        x = F.elu(self.gat_conv3(x, edge_index))
        x = F.elu(self.gat_conv4(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x

