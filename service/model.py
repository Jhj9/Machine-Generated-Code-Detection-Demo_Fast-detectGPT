import torch
from torch import nn
from transformers import AutoModel

class CNNLstmBert(nn.Module):
    def __init__(self, output_size):
        super(CNNLstmBert, self).__init__()
        self.bert1 = AutoModel.from_pretrained('ckiplab/bert-base-chinese')
        self.bert2 = AutoModel.from_pretrained('ckiplab/bert-base-chinese')

        self.cnn=nn.ModuleList()
        for i in range(5):
            cnns = nn.ModuleDict({
                f'conv1_{i}': nn.Conv2d(1, 768, kernel_size=(3, 768), stride=(1, 1), padding=(1, 0)),
                f'conv2_{i}': nn.Conv2d(1, 768, kernel_size=(3, 768), stride=(1, 1), padding=(1, 0))
            })
            self.cnn.append(cnns)

        self.lstm = nn.LSTM(768, 768, num_layers=3, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(768*3, output_size)
        self.dropout = nn.Dropout(0.3)


    def forward(self, x, device):
        # Word-level features
        embedding = self.bert1.embeddings(input_ids=x, position_ids=None,
                                         token_type_ids=torch.zeros(x.size(), dtype=torch.long, device=device),
                                         inputs_embeds=None)

        skip_connection = embedding.unsqueeze(1)
        conv_out = embedding.unsqueeze(1)
        for i, conv in enumerate(self.cnn):
            a = conv[f'conv1_{i}'](conv_out)
            b = conv[f'conv2_{i}'](conv_out)
            conv_out = a * torch.sigmoid(b)
            conv_out = torch.cat([conv_out[:, i, :, :] for i in range(768)], dim=-1)
            conv_out = conv_out.unsqueeze(1)
            conv_out = skip_connection+conv_out
            skip_connection = conv_out

        word_feature, _ = self.lstm(conv_out.squeeze(1))

        # Char-level features
        char_feature = self.bert2(x)[0]

        feature = torch.cat([word_feature, char_feature], dim=-1)
        x = feature.view(-1, feature.shape[2])
        x = self.fc(self.dropout(x))

        return feature, x