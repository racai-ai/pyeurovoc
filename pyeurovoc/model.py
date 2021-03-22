import torch.nn as nn


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, num_classes):
        super(LangModelWithDense, self).__init__()
        self.lang_model = lang_model

        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        cls_embedding = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        logits = self.dropout(self.linear(cls_embedding))

        return logits