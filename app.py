
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from tokenizer import Tokenizer

jsondata = []
max_lines = 2000000
hidden_size = 256
num_layers = 6
learning_rate = 0.001
num_epochs = 1
batch_size = 64
block_size = 48


# Read the input text
#with open('00.jsonl', 'r', encoding='utf-8') as f:
#    for _ in range(max_lines):
#        jsondata.append(f.readline())
#    #jsondata = f.readlines()
#
#text = ''
#
#for j in range(len(jsondata)):
#    text += json.loads(jsondata[0])['text']
#    jsondata.pop()
#    if j % 100 == 0:
#        print(f'Transfer: [{len(jsondata)} -> {j}]  ', end='\r')
#print(f'Transfer: [{len(jsondata)} -> {j}]  ', end='\r')
#
#len(text)
#
#del(jsondata)
#
#
#def write_large_data(data, file_path):
#    chunk_size = 4096  # Set the chunk size (adjust as needed)
#
#    with open(file_path, 'w+') as file:
#        for chunk in data_chunks(data, chunk_size):
#            file.write(chunk)
#
#
#def data_chunks(data, chunk_size):
#    for i in range(0, len(data), chunk_size):
#        yield data[i:i+chunk_size]
#
#write_large_data(text, 'dataset_3.txt')


# Read the input text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in characters:", len(text))

# Get all unique characters in the text
#chars = sorted(list(set(text)))
#vocab_size = len(chars)
#print(''.join(chars))
#print("Vocabulary size:", vocab_size)
#
## Create a mapping from characters to integers
#stoi = {ch: i for i, ch in enumerate(chars)}
#itos = {i: ch for i, ch in enumerate(chars)}
#encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
#decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
#
#print(encode("hii there"))
#print(decode(encode("hii there")))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = Tokenizer()
tokenizer.load(text)
vocab_size = len(tokenizer.mapping)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
print("Data shape:", data.shape, "Data dtype:", data.dtype)
print("Vocabulary size:", vocab_size)

# Split the data into train and validation sets
n = int(0.9 * len(data))  # First 90% will be train, rest val
print('N', n)
train_data = data[:n]
val_data = data[n:]

# Define the LSTM-based GPT-like model
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, hidden_size, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(num_layers, batch_size, hidden_size).zero_().to(device),
                  weight.new(num_layers, batch_size, hidden_size).zero_().to(device))
        return hidden


# Initialize the model
model = GPT(vocab_size, block_size, hidden_size, num_layers)
#model.load_state_dict(torch.load('gpt_2.pth'))

model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Count the number of parameters
num_params = sum(p.numel() for p in model.parameters())

# Convert the number of parameters to millions
num_params_in_millions = num_params / 1e6


# Half data.
#model = model.half()
#train_data = train_data.to(torch.int16).to(torch.long)


# Training loop
for epoch in range(num_epochs):
    # Set the model in training mode
    model.train()

    # Initialize the hidden state
    hidden = model.init_hidden(batch_size)

    td = tqdm(range(0, ((train_data.size(0)-1) - block_size) - (block_size*batch_size)), desc="Processing", dynamic_ncols=True)

    for i in td:
        # Prepare the inputs and targets for the current batch
        inputs_batch = train_data[i:i + block_size * batch_size].view(batch_size, block_size).to(device)
        targets_batch = train_data[i + 1:i + block_size * batch_size + 1].view(batch_size, block_size).to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, hidden = model(inputs_batch, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets_batch.view(-1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Detach hidden state to prevent backpropagation through time
        hidden = tuple(h.detach() for h in hidden)

        #if i % 500 == 0:
        #    print(f"Epoch [{epoch+1}/{num_epochs}], Params: {num_params_in_millions:.2f}M, Loss: {loss.item():.4f}, Itter: [{i}/{((train_data.size(0)-1) - block_size) - (block_size*batch_size)}], VRAM: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}MB, ", end='\r')
        tqdm.set_description(td, desc=f'{loss.item():.4f}')

    torch.save(model.state_dict(), 'current.pth')
    print()

torch.save(model.state_dict(), 'complete.pth')

# Set the model in evaluation mode
model.eval()

# Generate some sample text
start_sequence = '''Timmy: Hello, I am a robot
Billy: Thats so cool!
Timmy: '''
hidden = model.init_hidden(1)

with torch.no_grad():
    input_sequence = torch.tensor(tokenizer.encode(start_sequence), dtype=torch.long).unsqueeze(0).to(device)

    # Initialize the output sequence with the input sequence
    output_sequence = input_sequence

    # Generate the rest of the sequence
    for _ in range(256):
        output, hidden = model(input_sequence, hidden)
        output = output[:, -1, :]
        _, topk = torch.topk(output, 1)
        input_sequence = topk.squeeze(0).unsqueeze(0)
        output_sequence = torch.cat((output_sequence, input_sequence), dim=1)

    generated_text = tokenizer.decode(output_sequence.squeeze().tolist())
    print("Generated Text:")
    print(generated_text)
    print(list(generated_text))





