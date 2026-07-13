import torch
torch.autograd.set_detect_anomaly(True)

dropout_mlp = 0.5
dropout_gru = 0.25

class SingleMLP_Classifier(torch.nn.Module):
    def __init__(self, input_shape, dropout = dropout_mlp):
        super().__init__()
        self.dropout = dropout
        
        self.linear_relu_stack =torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(256, 2)
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class RNN_Classifier(torch.nn.Module):
    def __init__(self, dropout=dropout_gru):
        super().__init__()
        hidden_dim = 128
        num_layers = 4
        self.gru = torch.nn.GRU(1, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, seq):
        gru_out, _ = self.gru(seq)
        return self.linear(gru_out)[-1, -1, :]


class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        
        out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class DNN_Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(DNN_Classifier, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()

        self.res_block = ResidualBlock(hidden_size)

        self.layer3 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.res_block(x)

        x = self.layer3(x)
        
        return x
    

class Transformer_Residual_Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, nhead=8, dropout=dropout_mlp):
        super(Transformer_Residual_Classifier, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.transformer_block = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.res_block = ResidualBlock(hidden_size)
        self.layer4 = torch.nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = x.unsqueeze(1)
        x = self.transformer_block(x)
        x = x.squeeze(1)
        
        x = self.res_block(x)
        x = self.layer4(x)
        
        return x