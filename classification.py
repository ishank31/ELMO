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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch.nn.functional as F
nltk.download('punkt')
plt.style.use('ggplot')

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

data = list(df['Description'])
test_data = list(df_test['Description'])

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
cleaned_test_data = preprocess_text(test_data)

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

# Initialize a counter for word occurrences
min_count = 5
word_counts = Counter()

# Iterate through each sublist in the list of lists
for sublist in cleaned_test_data:
    # Update word counts with words from each sublist
    word_counts.update(sublist)

# Replace words with count less than min_count with <UNK> and add <BOS> and <EOS> tags
preprocessed_list_of_lists_test = []
for sublist in cleaned_test_data:
    # Replace words with count less than min_count with <UNK>
    preprocessed_sublist = ['<unk>' if word_counts[word] < min_count else word for word in sublist]

    # Add <BOS> and <EOS> tags to each sublist
    # padded_sublist = ['<BOS>'] * window_size + preprocessed_sublist + ['<EOS>'] * window_size
    preprocessed_list_of_lists_test.append(preprocessed_sublist)

df['Class Index'] -= 1
df_test['Class Index'] -= 1

labels = list(df['Class Index'])
test_labels = list(df_test['Class Index'])

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


sent_train_data = SentDataset(preprocessed_list_of_lists)
sent_train_loader = DataLoader(sent_train_data, batch_size=16, shuffle=True, collate_fn=sent_train_data.collate, num_workers = 10)

vocab = sent_train_data.vocabulary.get_stoi()
vocab_size = len(vocab)

class ClassifierDataset(Dataset):
    def __init__(self, data: list[list[str]], labels: list[int], vocabulary:Vocab|None=None):
        self.sentences = data
        self.labels = labels

        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get the datapoint at `index`."""
        sentence_tensor = torch.tensor(self.vocabulary.lookup_indices(self.sentences[index]))
        label = self.labels[index]
        return sentence_tensor, label

    def collate(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together."""
        sentences, labels = zip(*batch)
        padded_seq =  pad_sequence(sentences, batch_first=True, padding_value=self.vocabulary[PAD_TOKEN])
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        return padded_seq, label_tensor
    

train_data = ClassifierDataset(preprocessed_list_of_lists, labels)
train_loader = DataLoader(train_data, batch_size=16, collate_fn=train_data.collate, num_workers = 10, shuffle=True)

test_data = ClassifierDataset(preprocessed_list_of_lists_test, test_labels, vocabulary=train_data.vocabulary)
test_loader = DataLoader(test_data, batch_size=16, collate_fn=test_data.collate, num_workers = 10, shuffle=False)

vocabulary = train_data.vocabulary.get_stoi()

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
    

class ELMoClassifier(nn.Module):
    def __init__(self, sent_train_loader, train_loader, vocabulary, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_embeddings=None, freeze_embeddings=True, train_weights=True, dropout=0.5, learnable_function='linear'):
        super(ELMoClassifier, self).__init__()
        
        # Initialize the ELMo model
        self.elmo = ELMo(vocab_size, embedding_dim, hidden_dim, pretrained_embeddings)
        
        # Load the pre-trained ELMo weights
        elmo_state_dict = torch.load('elmo_pretrained_bidirection_20_epochs.pt')
        self.elmo.load_state_dict(elmo_state_dict)

        for param in self.elmo.parameters():
            param.requires_grad = False

        # Freeze the pre-trained embeddings (optional)
        if freeze_embeddings:
            self.elmo.embedding.weight.requires_grad = False

        delattr(self.elmo, 'linear')
        
        # Trainable scalar weights for combining the layers
        self.scalar_weights = nn.Parameter(torch.randn(3)) if train_weights else torch.Tensor([0.33, 0.33, 0.33])
        print("scalar weights: ", self.scalar_weights)
        self.train_weights = train_weights
        
        # Classifier layer for the downstream task
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(hidden_dim, hidden_dim * 2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_dim * 2, num_classes)
        self.num_classes = num_classes
        self.sent_train_loader = sent_train_loader
        self.classify_train_loader = train_loader

        self.vocabulary = vocabulary
        self.set_learnable_function(learnable_function)

        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_classes)
        self.relu = nn.ReLU()
        

    def set_learnable_function(self, learnable_function):
        if learnable_function == 'relu':
            self.learnable_function = torch.nn.ReLU()
        elif learnable_function == 'tanh':
            self.learnable_function = torch.nn.Tanh()
        elif learnable_function == 'sigmoid':
            self.learnable_function = torch.nn.Sigmoid()
        elif learnable_function == 'linear':
            self.learnable_function = lambda x: x 

    def forward(self, input_ids):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        embed = self.elmo.embedding(input_ids)
    
        # Pass through the pre-trained Bi-LSTM layers
        lstm1_output, _ = self.elmo.lstm1(embed)
        lstm2_output, _ = self.elmo.lstm2(lstm1_output)
    
        # Combine the layers using the scalar weights
        layer0 = lstm1_output
        layer1 = lstm2_output
        layer2 = embed
    
        if self.train_weights:
            combined_output = self.scalar_weights[0] * layer0 +\
                self.scalar_weights[1] * layer1 +\
                self.scalar_weights[2] * layer2
            
        else:
            combined_output = self.scalar_weights[0] * layer0.detach() +\
                self.scalar_weights[1] * layer1.detach() +\
                self.scalar_weights[2] * layer2.detach()
            
        # print("combined outptu shape: ", combined_output.shape)
        # combined_output = self.learnable_function(combined_output)

        lstm_output, _ = self.lstm(combined_output)
        final_hidden_state = lstm_output[:, -1, :]
        final_hidden_state = self.dropout(final_hidden_state)

        # logits = self.fc1(final_hidden_state)
        # logits = self.relu(logits)
        # logits = self.fc2(logits)
        # logits = self.relu(logits)
        # logits = self.fc3(logits)

        logits = self.decoder(final_hidden_state)
    
        return logits
    
    def train_classifier(self, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        self.losses = []
        self.accs = []

        for epoch in range(num_epochs):
            self.train()
            total_loss_epoch = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(self.classify_train_loader):
                optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = self(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss_epoch += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = total_loss_epoch / len(self.classify_train_loader)
            epoch_accuracy = total_correct / total_samples
            
            self.losses.append(epoch_loss)
            self.accs.append(epoch_accuracy*100)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy*100}%")

    def plot_accs(self):
        plt.plot(range(len(self.accs)), self.accs)
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.title("ELMo classifier Accuracy vs Epoch")
        plt.show()

    def plot_losses(self):
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.title("ELMo classifier Loss vs Epoch")
        plt.show()


    def test_classifier(self, test_loader, criterion):
        self.eval()  # Set the model to evaluation mode
        device = next(self.parameters()).device
        test_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():  # No need to compute gradients during inference
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
                test_loss += loss.item()  # Accumulate the loss
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Store labels and predictions for classification report and confusion matrix
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Calculate average loss and accuracy
        average_loss = test_loss / len(test_loader)
        accuracy = total_correct / total_samples
        
        return average_loss, accuracy, all_labels, all_predictions

    def generate_classification_report_and_confusion_matrix(self, all_labels, all_predictions, target_names):
        report = classification_report(all_labels, all_predictions, target_names=target_names)
        matrix = confusion_matrix(all_labels, all_predictions)
        return report, matrix
    
    def save_model(self, model_path):
        # Save model state dictionary
        torch.save(self.state_dict(), model_path) 



model = ELMoClassifier(sent_train_loader=sent_train_loader, train_loader=train_loader, vocabulary=vocabulary, vocab_size=vocab_size, embedding_dim=300, hidden_dim=300, num_classes=4, learnable_function='tanh', train_weights=True)

model.train_classifier(num_epochs=20)

model.save_model("classifier.pt")

criterion = nn.CrossEntropyLoss()
average_loss, accuracy, all_labels, all_predictions = model.test_classifier(test_loader=test_loader, criterion=criterion)

model.plot_accs()
model.plot_losses()
print(classification_report(all_labels, all_predictions))


conf_matrix = confusion_matrix(all_labels, all_predictions)
class_labels = ['Label 1', 'Label 2', 'Label 3', 'Label 4']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Confusion Matrix')
plt.show()