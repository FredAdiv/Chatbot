## Chatbot training module.

import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet


#Load sentiment data
with open('intents.json' , 'r') as f:
    intents = json.load(f)

all_words =[]
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) #We use extend because it's an array 
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']

#Implement Steming
all_words = [stem(w) for w in all_words if w not in ignore_words]

#Sort words
all_words = sorted(set(all_words)) #unique element
tags = sorted(set(tags))


#Training Data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #CrossEntropyLoss

#Convert to vectors
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    


#Configurations
batch_size = 8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#Check if GPU is Avaible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train
for epoch in range(num_epochs):
    for (words, label) in train_loader:
        words = words.to(device)
        label = label.to(device)

        #forward
        outputs = model(words)
        label = label.long() # Need to execute 
        loss = criterion(outputs,label)

        #backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item()}')
print(f' final loss, loss = {loss.item()}')


#Save the data in a pickle format
data = {
    'model_state' : model.state_dict(),
    'input_size' : input_size,
    'output_size' : output_size,
    'hidden_size' : hidden_size,
    'all_words' : all_words,
    'tags' : tags,
}

FILE = "trained.pth"
torch.save(data, FILE)

print(f'Training complete, file saved to {FILE}')