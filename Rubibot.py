
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ChatDataset(Dataset):
    def __init__(self, file_path):
        self.conversations = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                user_input = lines[i].strip()
                bot_response = lines[i+1].strip()
                self.conversations.append((user_input, bot_response))
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]

# Transformers
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(TransformerChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)
        output = self.transformer(src_embedded, tgt_embedded)
        output = self.linear(output)
        return output

# Chatbot
def chat_with_bot(model, vocab_size, device):
    bot_response = "Bot: Merhaba! Nasıl yardımcı olabilirim?"
    print(bot_response)

    while True:
        user_input = input("Sen: ")
        if user_input.lower() == 'quit':
            break
        
        user_input_encoded = torch.tensor([text_to_indices(user_input, vocab_size)], dtype=torch.long).to(device)
        bot_response = generate_response(model, user_input_encoded, vocab_size, device)
        print("Bot:", bot_response)


def text_to_indices(text, vocab_size):
    # Basit bir metin kodlama
    indices = [ord(c) for c in text]
    indices = indices[:min(len(indices), vocab_size)]
    indices += [0] * (vocab_size - len(indices))
    return indices


def generate_response(model, user_input_encoded, vocab_size, device):
    bot_response = ""
    with torch.no_grad():
        for i in range(20):  # Maksimum 20 token için döngü
            bot_output = model(user_input_encoded, user_input_encoded)[:, -1, :]
            predicted_index = torch.argmax(bot_output, dim=1).item()
            if predicted_index == 0:  # End of sequence token
                break
            bot_response += chr(predicted_index)
            user_input_encoded = torch.cat((user_input_encoded, torch.tensor([[predicted_index]], dtype=torch.long).to(device)), dim=1)
    return bot_response

# Dataset
chat_dataset = ChatDataset("chat_data.txt")
vocab_size = 128  # ASCII karakter sayısı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = DataLoader(chat_dataset, batch_size=1, shuffle=True)

# Model
model = TransformerChatbot(vocab_size, 256, 8, 4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for conversation in data_loader:
        user_input, bot_response = conversation[0]
        user_input_encoded = torch.tensor([text_to_indices(user_input, vocab_size)], dtype=torch.long).to(device)
        bot_response_encoded = torch.tensor([text_to_indices(bot_response, vocab_size)], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(user_input_encoded, bot_response_encoded[:, :-1])  # Son hedef tokeni çıkar
        loss = criterion(output.transpose(1, 2), bot_response_encoded[:, 1:])  # CrossEntropyLoss için hedef tokenler
        loss.backward()
        optimizer.step()

# Interaction
chat_with_bot(model, vocab_size, device)
