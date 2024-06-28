import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from tqdm import tqdm
nltk.download('punkt')

df = pd.read_csv('train.csv')
data = list(df['Description'])

def preprocess_text(sentences):
    tokenized_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())  # Lowercasing

        cleaned_words = []
        for word in words:
            # Remove punctuation marks
            word = re.sub(r'[^\w\s\\]', '', word)
            # Split words separated by '\'
            word = word.split('\\')
            cleaned_words.extend(word)

        # Remove empty strings and short words
        # cleaned_words = [word for word in cleaned_words if len(word) > 1]
        cleaned_words = [word for word in cleaned_words if word != '' and word != 's']

        tokenized_sentences.append(cleaned_words)

    return tokenized_sentences

cleaned_data = preprocess_text(data)

# Initialize a counter for word occurrences
min_count = 5
word_counts = Counter()

# Iterate through each sublist in the list of lists
for sublist in cleaned_data:
    # Update word counts with words from each sublist
    word_counts.update(sublist)

# Replace words with count less than min_count with <UNK> and add <BOS> and <EOS> tags
preprocessed_list_of_lists = []
for sublist in cleaned_data:
    # Replace words with count less than min_count with <UNK>
    preprocessed_sublist = ['<unk>' if word_counts[word] < min_count else word for word in sublist]

    # Add <BOS> and <EOS> tags to each sublist
    # padded_sublist = ['<BOS>'] * window_size + preprocessed_sublist + ['<EOS>'] * window_size
    preprocessed_list_of_lists.append(preprocessed_sublist)


vocab = list(set([word for sublist in preprocessed_list_of_lists for word in sublist]))


def create_word_index_dicts(vocabulary):
    word_to_index = {}
    index_to_word = {}

    for index, word in enumerate(vocabulary):
        word_to_index[word] = index
        index_to_word[index] = word

    return word_to_index, index_to_word

word_to_index, index_to_word = create_word_index_dicts(vocab)

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

class SentDataset(Dataset):
    def __init__(self, data: list[list[str]], vocabulary: Vocab | None = None):
        self.sentences = data

        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the datapoint at `index`."""
        # Split the input sequence into forward and backward parts
        input_sentence = self.sentences[index]
        forward_sentence = input_sentence
        backward_sentence = list(reversed(input_sentence))

        # Convert words to indices using the vocabulary
        forward_indices = torch.tensor(self.vocabulary.lookup_indices(forward_sentence))
        backward_indices = torch.tensor(self.vocabulary.lookup_indices(backward_sentence))

        return forward_indices, backward_indices

    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together."""
        forward_batch = [item[0] for item in batch]
        backward_batch = [item[1] for item in batch]

        # Pad sequences for both forward and backward parts
        padded_forward = pad_sequence(forward_batch, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        padded_backward = pad_sequence(backward_batch, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])

        return padded_forward, padded_backward

train_data = SentDataset(preprocessed_list_of_lists)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate, num_workers = 15)
vocab = train_data.vocabulary.get_stoi()

vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, pretrained_embeddings=None, train_weights=True):
        super(ELMo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer (optionally initialize with pretrained embeddings)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=True)

        # Bi-LSTM layers
        self.lstm1 = nn.LSTM(self.embedding_dim, self.embedding_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.embedding_dim, self.embedding_dim //2, num_layers=1, batch_first=True, bidirectional=True)

        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
        # self.linear_layer = nn.Linear(hidden_dim , hidden_dim)

    def forward(self, input_ids):
        embed = self.embedding(input_ids)

        # Pass through the first Bi-LSTM layer
        lstm1_output, _ = self.lstm1(embed)

        # Pass through the second Bi-LSTM layer
        lstm2_output, _ = self.lstm2(lstm1_output)

        lin_output = self.linear(lstm2_output)
        return lin_output

elmo_model = ELMo(vocab_size=len(vocab), embedding_dim=300, hidden_dim=300)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(elmo_model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
elmo_model.to(device)

num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    elmo_model.train()
    total_loss_epoch = 0.0
    
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        inputs_forward, inputs_backward = batch
        inputs_forward, inputs_backward = inputs_forward.to(device), inputs_backward.to(device)
        
        # Forward pass for forward sequence
        outputs_forward = elmo_model(inputs_forward)
        loss_forward = criterion(outputs_forward.view(-1, vocab_size), inputs_forward.view(-1))
        
        # Backward pass for backward sequence
        outputs_backward = elmo_model(inputs_backward)
        loss_backward = criterion(outputs_backward.view(-1, vocab_size), inputs_backward.view(-1))
        
        # Total loss
        total_loss = loss_forward + loss_backward
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        total_loss_epoch += total_loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss_epoch}")


torch.save(elmo_model.state_dict(), 'bilstm.pt')
