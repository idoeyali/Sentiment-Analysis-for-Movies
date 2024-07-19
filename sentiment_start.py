import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

batch_size = 32
output_size = 2
hidden_size = 64  # to experiment with

run_recurrent = True  # else run Token-wise MLP
use_RNN = True  # otherwise GRU
only_atten = False  # otherwise MLP + atten
add_attention = False  # otherwise just regular MLP
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size, True)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        # Concatenate input and previous hidden state
        combined = torch.cat((x, hidden_state), dim=1)
        # Compute new hidden state
        new_hidden = self.sigmoid(self.in2hidden(combined))
        # Compute output
        output = self.hidden2out(hidden_state)
        return output, new_hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        self.update_gate_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.updated_state_weights = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, hidden_state), dim=1)
        # Compute update and reset gates
        update_gate = self.sigmoid(self.update_gate_weights(combined))
        reset_gate = self.sigmoid(self.reset_gate_weights(combined))
        # Compute the updated hidden state
        combined_reset = torch.cat((x, reset_gate * hidden_state), dim=1)
        updated_state = self.tanh(self.updated_state_weights(combined_reset))
        # Compute the new hidden state
        new_hidden = (1 - update_gate) * hidden_state + update_gate * updated_state
        # Compute output
        output = self.hidden2out(hidden_state)
        return output, new_hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, add_attention):
        super(ExMLP, self).__init__()
        self.add_attention = add_attention
        self.ReLU = torch.nn.ReLU()
        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        if self.add_attention:
            self.atten = ExRestSelfAtten(hidden_size, hidden_size, hidden_size)
        self.layer2 = MatMul(hidden_size, int(hidden_size / 2))
        self.layer3 = MatMul(int(hidden_size / 2), output_size)

    def name(self):
        if self.add_attention:
            return "MLP+Attention"
        return "MLP"

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        if self.add_attention:
            # Apply attention layer
            x, atten_weights = self.atten(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)
        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.out_proj = MatMul(hidden_size, output_size, use_bias=False)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2 * atten_size + 1, hidden_size))

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        # x_nei has an additional axis that corresponds to the offset
        x_nei += self.positional_encoding

        # Applying attention layer
        query = self.W_q(x).unsqueeze(2)  # (batch, seq_len, 1, hidden_size)
        keys = self.W_k(x_nei)  # (batch, seq_len, 2*atten_size+1, hidden_size)
        values = self.W_v(x_nei)  # (batch, seq_len, 2*atten_size+1, hidden_size)

        # Scaled dot-product attention
        scores = torch.matmul(query,
                              keys.transpose(-2, -1)) / self.sqrt_hidden_size  # (batch, seq_len, 1, 2*atten_size+1)
        atten_weights = self.softmax(scores)  # (batch, seq_len, 1, 2*atten_size+1)
        context = torch.matmul(atten_weights, values).squeeze(2)  # (batch, seq_len, hidden_size)

        # Final projection
        x = self.out_proj(context)  # (batch, seq_len, output_size)

        return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, true_label, pred_label):
    print(f"Review: {rev_text}")
    print(f"True Label: {1 - true_label}")
    print(f"The output: {1 - pred_label}")
    print("Word scores:")
    for i, word in enumerate(rev_text):
        print(f"{word}: [{sbs1[i]:.2f}, {sbs2[i]:.2f}]")


# select model to use

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size)
else:
    if only_atten:
        model = ExRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size, add_attention)

print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model = torch.load(model.name() + ".pth")
total_iters = 0


def test_model():
    # Run model on selected examples
    with torch.no_grad():
        for labels, reviews, reviews_text in test_dataset:
            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0]))
                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
            else:
                sub_score = model(reviews)
                output = torch.mean(sub_score, 1)

            # Calculate loss
            loss = criterion(output, labels)

            # Calculate predictions
            _, predicted = torch.max(output.data, 1)
            _, acc_labels = torch.max(labels.data, 1)
            # Iterate through the batch
            for i in range(labels.size(0)):
                review_text = reviews_text[i]
                print("############")
                print(i)
                print("############")
                print(f"True Label:{1 - acc_labels[i]}")
                print(f"The output:{1 - predicted[i]}")
                if not run_recurrent:
                    sbs1 = sub_score[i, :, 0].cpu().numpy()
                    sbs2 = sub_score[i, :, 1].cpu().numpy()
                    true_label = acc_labels[i].item()
                    pred_label = predicted[i].item()
                    # Print review details
                    print_review(review_text, sbs1, sbs2, true_label, pred_label)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
test_iters = []

# training steps in which a test step is executed every test_interval
for epoch in range(num_epochs):
    itr = 0  # iteration counter within each epoch
    for labels, reviews, reviews_text in train_dataset:  # getting training batches
        itr = itr + 1
        total_iters += 1
        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch

        else:
            test_iter = False
        # Recurrent nets (RNN/GRU)
        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))
            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
        else:
            # Token-wise networks (MLP / MLP + Atten.)
            sub_score = []
            if only_atten:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)
            output = torch.mean(sub_score, 1)
        # cross-entropy loss
        loss = criterion(output, labels)
        # optimize in training iterations
        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # averaged losses
        # Inside the test iteration block
        if test_iter:
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
            # Compute test accuracy
            with torch.no_grad():
                _, predicted = torch.max(output.data, 1)
                _, acc_labels = torch.max(labels.data, 1)
                correct = (predicted == acc_labels).sum().item()
                test_accuracy = 100 * correct / labels.size(0)
                test_accuracies.append(test_accuracy)
                test_iters.append(total_iters)
            test_losses.append(test_loss)
            # if total_iters%300 ==0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}, "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )
            if not run_recurrent:
                nump_subs = sub_score.detach().numpy()
                labels = labels.detach().numpy()
                print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], predicted[0], acc_labels[0])
            # saving the model
            torch.save(model, model.name() + ".pth")
        else:
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss
            # Compute train accuracy
            _, predicted = torch.max(output.data, 1)
            _, acc_labels = torch.max(labels.data, 1)
            correct = (predicted == acc_labels).sum().item()
            train_accuracy = 100 * correct / labels.size(0)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
# If you only want to test pre-trained model
# test_model()

# Plotting
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_iters, test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
# plt.title(f'Train and Test Losses for {model.name()}\nHidden State Size = {hidden_size}')
plt.title(f'Train and Test Losses for {model.name()}')
plt.legend()
plt.grid(True)

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_iters, test_accuracies, label='Test Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy (%)')
# plt.title(f'Train and Test Accuracies for {model.name()}\nHidden State Size = {hidden_size}')
plt.title(f'Train and Test Accuracies for {model.name()}')
plt.legend()
plt.grid(True)

plt.show()
