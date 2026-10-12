"""Small text CNN and LSTM classifiers with padding handled explicitly."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    def __init__(self, vocab_size, classes=3, embedding_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.kernels = (3, 4, 5)
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, hidden_dim, k) for k in self.kernels)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim * len(self.kernels), classes)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens).transpose(1, 2)
        summaries = []
        for k, conv in zip(self.kernels, self.convs):
            activations = torch.relu(conv(embedded))
            valid_windows = (lengths - k + 1).clamp(min=1).to(tokens.device)
            positions = torch.arange(activations.shape[-1], device=tokens.device)
            invalid = positions[None, :] >= valid_windows[:, None]
            activations = activations.masked_fill(invalid[:, None, :], -torch.inf)
            summaries.append(activations.max(dim=-1).values)
        return self.output(self.dropout(torch.cat(summaries, dim=1)))


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, classes=3, embedding_dim=64, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, classes)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(self.dropout(hidden[-1]))
