## Chatbot application
import random
import json
from train import all_words, dataset, intents
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load sentiment data
with open('intents.json', 'r') as f:
    intents = json.load(f)

#Load saved model
FILE = "trained.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Ghost"

print("Hello, my name is ",bot_name , ". How can I help? type 'quit' to exit!")

while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]


    if prob.item() > 0.40: #We found a match with over 70% probability
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")