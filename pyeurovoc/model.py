import torch.nn as nn


class LangModelWithDense(nn.Module):
    def __init__(self, lang_model, num_classes):
        """
            BERT model class. We use a simple feed-forward layer to map the embedding of the first token into a
            probability distribution for each EuroVoc descriptor.

            Args:
                lang_model (str): Language model to be used according to HuggingFace.
                num_classes (int): Number of EuroVoc descriptors.
        """
        super(LangModelWithDense, self).__init__()
        self.lang_model = lang_model

        self.linear = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        """
            Function that computes the logits of an input tokens.

            Args:
                x (tensor): The input tensor of the tokenized text.
                mask (tensor): The mask of the input tokens.
            Returns:
                A tensor containing the logits of the probability distribution.
        """
        cls_embedding = self.lang_model(x, attention_mask=mask)[0][:, 0, :]

        logits = self.dropout(self.linear(cls_embedding))

        return logits