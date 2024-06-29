import tqdm
import json
import os
import gc
import torch
import torch.nn as nn
from torch import optim
from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertForMaskedLM#, AutoModelForSequenceClassification #, BertForSequenceClassification, AutoTokenizer
from torchmetrics import MetricCollection, F1Score, Accuracy, Precision, Recall, ConfusionMatrix
from torchmetrics.classification import MulticlassAUROC
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

class TextFixerBERTModel(nn.Module):
    
    def __init__(self, base_model_path=None):
        super(TextFixerBERTModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(base_model_path, output_hidden_states = True,)  
        self.bert_tokenizer = BertTokenizer.from_pretrained(base_model_path)
        vocab_size = len(self.bert_tokenizer)
        embedding_dim = self.bert.bert.embeddings.word_embeddings.embedding_dim
            
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)
        self.device = device

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeds = self.bert(input_ids=input_ids, attention_mask= attention_mask, labels=labels)

        return embeds

def train_bert_embeddings_masked(base_model_path, batch_size=1,  number_epochs=10, learning_rate=5e-5, n_bottom_layers_freeze=4, n_edge_freezed_layer=9):        
    print('Training with learning rate', learning_rate) 
    model = TextFixerBERTModel(base_model_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(0, number_epochs):
        global_step = 0        
        print('New epoch ', epoch) 
        preds_values_epoch = []
        labels_values_epoch = [] 
        historical_loss = []
        
        set_no_grad(model, n_bottom_layers_freeze, n_edge_freezed_layer)
        
        model.train()
        
        total_loss = 0  
        total_iterations =1
        loop = tqdm.tqdm(range(0,total_iterations, batch_size))
        
        for i_step in loop:
            loop.set_description(f'Train Epoch {epoch}')  
            global_step +=1
            batch_data = {}   
            for j in range(batch_size): 
                index_current = i_step + j 
                context = 'Here is the initial text'#training_set[index_current]['context']
                term = 'Here is the initial text' #training_set[index_current]['term']
                batch_data = prepare_input(context, 0.2, model.bert_tokenizer, batch_data)

            model.zero_grad()
            try:
                t_input_ids = torch.tensor(batch_data['input_ids'], dtype=torch.long)
                t_attention_masks = torch.tensor(batch_data['attention_mask'], dtype=torch.long)
                t_labels = torch.tensor(batch_data['labels'], dtype=torch.long)
                log_probs = model(input_ids=t_input_ids.to(model.device), attention_mask=t_attention_masks.to(model.device), labels=t_labels.to(model.device)) 
                loss = log_probs.loss
                
                loss.backward() 
                optimizer.step()
                current_loss = loss.cpu().item() 
                total_loss +=  current_loss
                loop.set_postfix(loss=current_loss)
                historical_loss.append(total_loss)
                predictions_batch = log_probs.logits.cpu().argmax(dim=2).numpy()
                
                for i_sample, sample in enumerate(batch_data['labels']):
                    samples_idx = [i for i, item in enumerate(sample) if item >0] 
                    y_value = [sample[idx] for idx in samples_idx]
                    pred_value = [predictions_batch[i_sample][idx] for idx in samples_idx]
                    labels_values_epoch.extend(y_value)
                    preds_values_epoch.extend(pred_value)                       
                    
            except Exception as err:
                print(err)  
                print('************************Inputs IDS**********************************')
                print(batch_data['input_ids'])
                print('************************Loss**********************************')
                print(loss)
                continue
        
              
        
import random
def prepare_input(context, perc_mask, tokenizer, batch_data, max_size = 512):
    text_tokens = tokenizer.tokenize(context)
    total_masked_tokens = math.ceil(len(text_tokens)*perc_mask)
    rand_indexes = {}
    while len(rand_indexes)<total_masked_tokens:
        index = random.randint(0,len(text_tokens)-1)
        rand_indexes[index] = True

    input_data = tokenizer.encode_plus(context) 
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer._mask_token)
    input_data['input_ids'] =[mask_token_id if rand_indexes.get(i-1) is not None else item for i, item in enumerate(input_data['input_ids']) ]
    text_tokens = [tokenizer._mask_token if rand_indexes.get(i) is not None else item for i, item in enumerate(text_tokens) ]

    labels_tokens = [tokenizer._pad_token] + text_tokens + [tokenizer._pad_token]
    input_data.labels = [item if item >0 else -100 for item in tokenizer.convert_tokens_to_ids(labels_tokens)]
    

    if  batch_data.get('input_ids') is None:
        batch_data['input_ids'] = []  
        batch_data['attention_mask'] = []  
        batch_data['labels'] = []  
    
    batch_data['input_ids'].append(input_data.input_ids)  
    batch_data['attention_mask'].append(input_data.attention_mask)
    batch_data['labels'].append(input_data.labels)
        
    return batch_data


def set_no_grad(model, n_bottom_layers_freeze, n_edge_freezed_layer): 
    if n_bottom_layers_freeze <0 or n_bottom_layers_freeze > 8:
        raise Exception('The number of freezed layers is invalid (0-9):', n_bottom_layers_freeze)
    if n_edge_freezed_layer < n_bottom_layers_freeze+1 or n_edge_freezed_layer > 9:
        raise Exception(f'The edge freezed layer should be bigger than {n_bottom_layers_freeze+1} and below than 10')
    
    print('Freezing encoders layer from 0 to', n_bottom_layers_freeze, 'out of',len(model.bert.bert.encoder.layer), 'and layer {n_edge_freezed_layer}')

    if n_bottom_layers_freeze >0 and n_bottom_layers_freeze < 12:
        for param in model.bert.parameters(): #encoder and embeddings
            param.requires_grad = False
    
        print('Freezing the first:', n_bottom_layers_freeze, 'layers of BERT.')
        
        for i_layer in range(min(n_bottom_layers_freeze, 12), 12):
            for param in model.bert.bert.encoder.layer[i_layer].parameters():
                param.requires_grad = True

if __name__ == '__main__':
    base_model_path = 'google-bert/bert-base-multilingual-uncased'
    train_bert_embeddings_masked(base_model_path,  number_epochs=1)        
    