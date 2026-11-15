import torch
torch.autograd.set_detect_anomaly(True)


class SingleMLP_Classifier(torch.nn.Module):
    def __init__(self, input_shape, dropout=0.25):
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
    def __init__(self, dropout=0.25):
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
    def __init__(self, input_size, hidden_size=256, nhead=8, dropout=0.25, transformer_layers=3):
        super(Transformer_Residual_Classifier, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_block = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=transformer_layers
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


class SequenceLengthReducer(torch.nn.Module):
    """Reduce a variable-length hidden-state sequence to one feature vector."""

    def __init__(self, max_seq_len, reduced_len_1, reduced_len_2, dropout=0.2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.reducer = torch.nn.Sequential(
            torch.nn.Linear(max_seq_len, reduced_len_1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(reduced_len_1, reduced_len_2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(reduced_len_2, 1),
        )

    def forward(self, hidden_states):
        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected hidden states with shape "
                "[batch_size, sequence_length, hidden_dim]."
            )
        if hidden_states.shape[1] != self.max_seq_len:
            raise ValueError(
                f"Expected sequence length {self.max_seq_len}, "
                f"received {hidden_states.shape[1]}."
            )
        return self.reducer(hidden_states.transpose(1, 2)).squeeze(-1)


class HallucinationClassifier(torch.nn.Module):
    """Sequence reducer followed by the shared MLP classifier."""

    def __init__(
        self,
        hidden_dim,
        max_seq_len,
        reduced_len_1,
        reduced_len_2,
        reducer_dropout=0.2,
        mlp_dropout=0.2,
    ):
        super().__init__()
        self.sequence_reducer = SequenceLengthReducer(
            max_seq_len=max_seq_len,
            reduced_len_1=reduced_len_1,
            reduced_len_2=reduced_len_2,
            dropout=reducer_dropout,
        )
        self.classifier = SingleMLP_Classifier(
            input_shape=hidden_dim,
            dropout=mlp_dropout,
        )

    def forward(self, hidden_states):
        return self.classifier(self.sequence_reducer(hidden_states))


# New names for new code; the old names remain available to legacy scripts.
SingleMLPClassifier = SingleMLP_Classifier
RNNClassifier = RNN_Classifier
DNNClassifier = DNN_Classifier
TransformerResidualClassifier = Transformer_Residual_Classifier
