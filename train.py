import torch
import spacy
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import spacy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
df = pd.read_csv("dataset2.csv")
class SimilarityModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', context_embedding_size=300):
        super(SimilarityModel, self).__init__() 
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert.dropout = nn.Identity() 
        self.fc1 = nn.Linear(768 * 2 + context_embedding_size, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, context): 
        output_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2) 
        pooled_output_1 = output_1.pooler_output
        pooled_output_2 = output_2.pooler_output 
        combined_input = torch.cat((pooled_output_1, pooled_output_2, context), dim=1) 
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x  
class SimilarityDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.nlp = spacy.load("en_core_web_md") 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence_1 = self.data.iloc[idx]['Sentence 1']
        sentence_2 = self.data.iloc[idx]['Sentence 2']
        context = self.data.iloc[idx]['Context']
        similarity = self.data.iloc[idx]['Similarity'] 
        encoding_1 = self.tokenizer(sentence_1, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        encoding_2 = self.tokenizer(sentence_2, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt') 
        context_embedding = self.get_context_embedding(context)
        
        return {
            'input_ids_1': encoding_1['input_ids'].squeeze(0),
            'input_ids_2': encoding_2['input_ids'].squeeze(0),
            'attention_mask_1': encoding_1['attention_mask'].squeeze(0),
            'attention_mask_2': encoding_2['attention_mask'].squeeze(0),
            'context': context_embedding,
            'similarity': torch.tensor(similarity, dtype=torch.long)
        }
    
    def get_context_embedding(self, context):
        doc = self.nlp(context)
        context_embedding = torch.tensor(doc.vector, dtype=torch.float32) 
        return context_embedding
def train_model(model, train_dataset, val_dataset, batch_size=16, epochs=10, lr=1e-3): 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad() 
            input_ids_1 = batch['input_ids_1']
            attention_mask_1 = batch['attention_mask_1']
            input_ids_2 = batch['input_ids_2']
            attention_mask_2 = batch['attention_mask_2']
            context = batch['context']
            similarity = batch['similarity']
            
            output = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, context) 
            loss = loss_fn(output, similarity)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}') 
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids_1 = batch['input_ids_1']
                attention_mask_1 = batch['attention_mask_1']
                input_ids_2 = batch['input_ids_2']
                attention_mask_2 = batch['attention_mask_2']
                context = batch['context']
                similarity = batch['similarity']
                
                output = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, context)
                preds = torch.argmax(output, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(similarity.cpu().numpy()) 
        accuracy = accuracy_score(val_labels, val_preds)
        print(f'Validation Accuracy: {accuracy:.4f}') 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = SimilarityDataset(train_data, tokenizer)
val_dataset = SimilarityDataset(val_data, tokenizer) 
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) 
model = SimilarityModel() 
train_model(model, train_dataset, val_dataset)  
