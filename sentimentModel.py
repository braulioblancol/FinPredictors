import tqdm
import json
import copy
import os
import gc
import torch
import torch.nn as nn
from torch import optim
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AutoModelForSequenceClassification #, BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle
import statistics
import math

class Object(object):
    pass

class CustomBERTModel(nn.Module):
    
    def __init__(self, number_labels = 2, device=None, model_type='SA', dropout_perc = 0.1, is_bert_lstm=None, hidden_layers=5, lstm_sequence_length=15, stacked_lstm_layers=1, tokenizer=None, local_model_location = None, dense_model_type=0, lambda1 = 512, lambda2 = 256):
        super(CustomBERTModel, self).__init__()
        if device is None:
            return
        self.args = {'number_labels':number_labels,'device':device,'dropout_perc':dropout_perc, 'model_type': model_type, 'is_bert_lstm': is_bert_lstm, 'hidden_layers':hidden_layers, 'lstm_sequence_length':lstm_sequence_length,'stacked_lstm_layers':stacked_lstm_layers,'tokenizer':tokenizer, 'local_model_location':local_model_location,'dense_model_type':dense_model_type}
        self.model_type = model_type
        self.num_labels = number_labels
        self.lstm_sequence_length = lstm_sequence_length
        self.lstm_hidden_layers = hidden_layers
        self.stacked_lstm_layers = stacked_lstm_layers

        if local_model_location is not None: 
            if local_model_location == 'FinBERT':
                self.bert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",num_labels=number_labels, ignore_mismatched_sizes=True)
            else:
                bert_masked = BertForMaskedLM.from_pretrained(local_model_location, output_hidden_states = True,)   
                self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased", output_hidden_states = True,)
                self.bert.resize_token_embeddings(len(tokenizer))
                for target_embeddings, weights_masked in zip(self.bert.embeddings.parameters(),bert_masked.bert.embeddings.parameters()):
                    target_embeddings.data.copy_(weights_masked.clone())
                print("Weights cloned from tokenizer's embedding")
                bert_masked = None
        else:
            self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased", output_hidden_states = True,)
            if tokenizer is not None and len(tokenizer) != self.bert.embeddings.word_embeddings.num_embeddings:
                diff_tokens = len(tokenizer)-self.bert.embeddings.word_embeddings.num_embeddings
                print('Resizing bert embeddings from ',self.bert.embeddings.word_embeddings.num_embeddings, 'to',len(tokenizer), 'delta:', diff_tokens)
                self.bert.resize_token_embeddings(len(tokenizer))


        print('Dropout', dropout_perc)
       
        self.bert.config.hidden_dropout_prob = dropout_perc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert.to(device)
        self.dropout_perc = dropout_perc
        self.is_bert_lstm = is_bert_lstm
        self.is_finbert = False
        self.device = device
        self.dense_model_type = dense_model_type


        if local_model_location == 'FinBERT':
            self.is_finbert = True
        else:
            self.dropout = nn.Dropout(self.dropout_perc) 
            if self.is_bert_lstm:
                print('dense_model_type',dense_model_type)
                if  dense_model_type == 0:
                    self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_layers, num_layers=1, bias=True, batch_first=True, dropout=self.dropout_perc)
                    self.lstm.to(device)
                    self.classifier = nn.Linear(hidden_layers, self.num_labels)
                elif dense_model_type ==1 or  dense_model_type ==3:
                    self.dense1 = nn.Linear(self.bert.config.hidden_size, lambda1)
                    self.dense1.to(device)
                    self.dense2 = nn.Linear(lambda1, lambda2)
                    self.dense2.to(device)
                    self.lstm = nn.LSTM(input_size=lambda2,  hidden_size=hidden_layers, num_layers=1, bias=True, batch_first=True, dropout=self.dropout_perc)
                    self.lstm.to(device)
                    if dense_model_type ==1:
                        if stacked_lstm_layers >1:
                            self.lstm2 = nn.LSTM(input_size=hidden_layers, hidden_size=hidden_layers, num_layers=1, bias=True, batch_first=True, dropout=self.dropout_perc)
                            self.lstm2.to(device)
                        self.classifier = nn.Linear(hidden_layers, self.num_labels)
                    else:
                        self.dense3 = nn.Linear(hidden_layers*stacked_lstm_layers, lambda1)
                        self.dense4 = nn.Linear(lambda1, lambda2)
                        self.classifier = nn.Linear(lambda2, self.num_labels)

                elif dense_model_type==2:
                    self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_layers, num_layers=1, bias=True, batch_first=True, dropout=self.dropout_perc)
                    self.lstm.to(device)
                    self.dense1 = nn.Linear(hidden_layers, lambda2)
                    self.classifier = nn.Linear(lambda2, self.num_labels)
                elif dense_model_type==4 or dense_model_type==5:
                    if dense_model_type==4:
                        self.dense1 = nn.Linear(self.bert.config.hidden_size, lambda1)
                        self.dense1.to(device)
                        self.dense2 = nn.Linear(lambda1, lambda2)
                        self.dense2.to(device)
                        concat_docs_length=lambda2*lstm_sequence_length
                        lambda3 = int(concat_docs_length / 2)
                        self.dense3 = nn.Linear(concat_docs_length, lambda3)
                        self.dense3.to(device)
                    else:
                        concat_docs_length=self.bert.config.hidden_size*lstm_sequence_length
                        lambda3 = int(concat_docs_length / 2)
                        self.dense3 = nn.Linear(concat_docs_length, lambda3)
                        self.dense3.to(device)

                    lambda4 = pow(2,math.floor(math.log2(concat_docs_length))-1)
                    self.dense4 = nn.Linear(lambda3, lambda4)
                    self.dense4.to(device)
                    if stacked_lstm_layers >1:
                        concat_years_length = lambda4*stacked_lstm_layers
                        lambda5 = pow(2,math.floor(math.log2(concat_years_length))-1)
                        self.dense5 = nn.Linear(concat_years_length, lambda5)
                        self.dense5.to(device)
                        lambda6 = int(lambda5/2)
                        self.dense6 = nn.Linear(lambda5, lambda6)
                        self.dense6.to(device)
                        self.classifier = nn.Linear(lambda6, self.num_labels)
                    else:
                        lambda5 = int(lambda4/2)
                        self.dense5 = nn.Linear(lambda4, lambda5)
                        self.classifier = nn.Linear(lambda5, self.num_labels)
            else:
                self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
            self.classifier.to(device)

    def init_hidden_lstm(self, batch_size):
        print('Reinitialization of hidden lstm')
        h0 = torch.zeros((self.stacked_lstm_layers, batch_size, self.lstm_hidden_layers)).detach().to(self.device)
        c0 = torch.zeros((self.stacked_lstm_layers, batch_size, self.lstm_hidden_layers)).detach().to(self.device)
        hidden = (h0,c0)
        return hidden

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids, 
            return_logits=None,
            return_dict=None,
            labels=None,  
            weights=None,
            lstm_hidden=None,
            output_hidden_states = None
    ):
        #with torch.set_grad_enabled(True):
        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict
        input_ids = input_ids.long() 
        embeds = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
            return_dict=return_dict,
        )
        
        #hidden_states = embeds.hidden_states
        #attentions = embeds.attentions

        if self.is_finbert:
            logits = embeds[0]
            labels_ = labels
        else:
            sequence_output = embeds[1] #[0]:entire output, token by token ,[1]: output only for the first token [CLS]
            
            #if return_dict:
            #    del embeds

            sequence_output = self.dropout(sequence_output)
            #sequence_output.retain_grad()
            if self.is_bert_lstm:
                batch_size = int(sequence_output.shape[0]/(self.lstm_sequence_length*self.stacked_lstm_layers))
                try:
                    sequence_output = torch.reshape(sequence_output, [batch_size,self.lstm_sequence_length*self.stacked_lstm_layers,sequence_output.shape[-1]]) #torch.unsqueeze(sequence_output, 0)
                except Exception as ex:
                    print('Sequence output', sequence_output.shape)
                    print('Input ids:', input_ids.shape)
                    print(ex)
                    raise ex

                if self.dense_model_type ==0:
                    _, (hidden_last,_) =  self.lstm(sequence_output, lstm_hidden) # hidden 
                    hidden_last_last_layer = hidden_last[-1] # (num_layers, N [batch_size], H_out [hidden_size])
                    logits = self.classifier(hidden_last_last_layer)
                elif self.dense_model_type==1 or self.dense_model_type==3:
                    dense_output = self.dense1(sequence_output)
                    dense_output = self.dense2(dense_output)
                    _, (hidden_last,_) = self.lstm(dense_output.view(batch_size*self.stacked_lstm_layers, self.lstm_sequence_length, dense_output.shape[-1]), lstm_hidden)
                    if self.dense_model_type==1:
                        if self.stacked_lstm_layers>1:
                            _, (hidden_last,_) =  self.lstm2(hidden_last.view(batch_size,self.stacked_lstm_layers,hidden_last.shape[-1]), lstm_hidden) # hidden 
                        hidden_last_last_layer = hidden_last[-1] # (num_layers, N [batch_size], H_out [hidden_size])
                    else:                        
                        dense_output2 = self.dense3(hidden_last.view(batch_size,hidden_last.shape[-1]*self.stacked_lstm_layers))
                        hidden_last_last_layer = self.dense4(dense_output2)
                    logits = self.classifier(hidden_last_last_layer)
                elif self.dense_model_type ==4 or self.dense_model_type ==5:
                    if self.dense_model_type ==4:
                        dense_output = self.dense1(sequence_output)
                        dense_output = self.dense2(dense_output)
                        concat_seq_output = self.dense3(dense_output.view(batch_size,self.stacked_lstm_layers,dense_output.shape[-1]*self.lstm_sequence_length))
                    else:
                        concat_seq_output = self.dense3(sequence_output.view(batch_size,self.stacked_lstm_layers,sequence_output.shape[-1]*self.lstm_sequence_length))

                    dense_output = self.dense4(concat_seq_output)
                    if self.stacked_lstm_layers >1:
                        dense_output = self.dense5(dense_output.view(batch_size,dense_output.shape[-1]*self.stacked_lstm_layers))
                        dense_output = self.dense6(dense_output)
                        logits = self.classifier(dense_output)                    
                    else:
                        dense_output = self.dense5(dense_output.view(batch_size,dense_output.shape[-1]*self.stacked_lstm_layers))
                        logits = self.classifier(dense_output)  
                else: 
                    _, (hidden_last,_) =  self.lstm(sequence_output, lstm_hidden) # hidden 
                    hidden_last_out = hidden_last[-1] # (num_layers, N [batch_size], H_out [hidden_size])
                    dense_output = self.dense1(hidden_last_out)
                    logits = self.classifier(dense_output)
            else:          
                logits = self.classifier(sequence_output) 
                #logits.retain_grad()
        #outputs = logits[(slice(None), (1,))]
        #torch.autograd.grad(torch.unbind(outputs), input_ids)

        labels_ = labels
        loss = None
        if labels_ is not None:
            if weights is not None:
                class_weights = torch.FloatTensor(weights).to(input_ids.device.type)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels_.view(-1)) 

        if return_logits:
            return logits

        if not return_dict:
            output = (logits,) + embeds[2:]
            return ((loss,) + output) if loss is not None else output

        if self.is_bert_lstm:
            return {
                'loss':loss,
                'logits':logits
            }
        else:
            return {
                'loss':loss,
                'logits':logits,
                #'hidden_states':hidden_states,
                #'attentions':attentions,
            }

    def from_pretrained(self, model_dir):
        args = torch.load(os.path.join(model_dir, "training_args.bin"))
        print('Loading from pre-trained model ', args)
        model_file = os.path.join(model_dir, "pytorch_model.bin")
        model = CustomBERTModel(**args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(os.path.join(model_dir,'pytorch_model.bin'), map_location=torch.device(device)))

        return model

    def save_pretrained(self, output_dir):
        assert os.path.isdir(output_dir)
        self.args['is_bert_lstm'] = self.is_bert_lstm

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

        #Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        print("Model weights saved in {}".format(output_model_file))

        #self.save_pretrained(save_directory=self.base_model_path) 

    def segment_dataset(self, dataset, segmentation_type = 'Default', segment_max_size=None, shift_size=None):
        new_dataset = []
        if segmentation_type =='topic':
            raise Exception('Segmentation type not yet implemented ' + segmentation_type)
        elif segmentation_type == 'paragraph':
            raise Exception('Segmentation type not yet implemented ' + segmentation_type)
        else: #Default  #Specifies the number of segments and the size of the shift
            for entry in dataset:
                number_segments = len(entry)/(segment_max_size-shift_size)
                segments = []
                for i_segment in range(number_segments):
                    i_start = i_segment*segment_max_size+shift_size
                    i_end = i_start + segment_max_size
                    segments.append(entry[i_start:i_end])
                new_dataset.append(segments)
        return new_dataset


def get_model(action, data_dir, output_dir, num_labels, device, lstm_args, model_type=None, tokenizer=None, dropout_perc=None,local_model_location=None,dense_model_type=0):
    model_dir = os.path.join(data_dir, output_dir)
    model = None
    last_epoch = -1
    if action == 'train':
        for file in os.listdir(model_dir):
            if 'pytorch_model.bin' == file:
                print('Loading model from pytorch_model.bin in ', model_dir)   
                model = CustomBERTModel().from_pretrained(model_dir=model_dir)
                file_name, results = read_metrics(model_dir, model_type, action)
                if len(results)>0:
                    last_epoch = results[-1]['last_epoch']
                break 
        if model is None:
            if local_model_location is None:
                local_model_location = "bert-base-multilingual-uncased"
                print('Loading bert-base-multilingual-uncased model')
            elif local_model_location.lower() == "finbert":
                local_model_location = 'FinBERT'
                print('Loading FinBERT model')
            else:
                print('Loding model from local directory',local_model_location)
            model = CustomBERTModel(device=device, number_labels=num_labels,is_bert_lstm= lstm_args.is_bert_lstm,hidden_layers= lstm_args.hidden_layers,lstm_sequence_length= lstm_args.lstm_sequence_length,stacked_lstm_layers=lstm_args.stacked_lstm_layers, tokenizer=tokenizer, dropout_perc=dropout_perc, local_model_location=local_model_location,dense_model_type=dense_model_type)
        model.train()
    elif action == 'test': 
        model = CustomBERTModel().from_pretrained( model_dir=model_dir)
        model.eval()
    model.to(device)
    return model, last_epoch

def drawLabelsDistributionPerStep(labels, batch_size,output_dir, file_name):
    class_distribution_template = {item:0 for item in labels}
    x_steps = np.arange(int(len(labels)/batch_size))
    y_values =  copy.deepcopy(class_distribution_template)
    y_values = {item:[] for item in y_values}
    for i_step in x_steps:
        class_distribution = copy.deepcopy(class_distribution_template)
        labels_batch = Counter(labels[i_step*batch_size:(i_step+1)*batch_size])
        for freq_label in labels_batch:
            class_distribution[freq_label] = labels_batch[freq_label]
        for class_ in class_distribution:
            y_values[class_].append(class_distribution[class_])

    if len(y_values) ==2:
        max_plots_x = 1
        max_plots_y = 2
    elif len(y_values) ==3:
        max_plots_x = 2
        max_plots_y = 2
    else:
        max_plots_x = int(len(y_values)/3)
        max_plots_y = 3

    if max_plots_x > 0 and max_plots_y>0:
        fig, axis = plt.subplots(max_plots_x, max_plots_y, figsize=(13,8))
        i = 0
        for x_plot  in range(max_plots_x):   
            for y_plot  in range(max_plots_y):       
                if len(y_values) ==2: 
                    axis[i].plot(x_steps, y_values[i])
                    axis[i].set_title("Label" + str(i))
                else:
                    if i< len(y_values): 
                        axis[x_plot, y_plot].plot(x_steps, y_values[i])
                        axis[x_plot, y_plot].set_title("Label" + str(i))
                i += 1

        fig.suptitle('Distribution of classes over steps') 
        plt.savefig(os.path.join(output_dir,file_name + '.jpg'))

def set_no_grad(model, number_layers_freeze, is_train):
    requires_grad = True if is_train else False
    for param in model.bert.parameters():
        param.requires_grad = requires_grad
    
    if number_layers_freeze >0 and is_train and 'finbert' not in model.bert.name_or_path:
        if number_layers_freeze > 12:
            print('Freezing all BERT layers.')
            for param in model.bert.parameters():
                param.requires_grad = False
        else:
            print('Freezing the first:', number_layers_freeze, 'layers of BERT.')
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
            for i_layer in range(min(number_layers_freeze, 12)):
                for param in model.bert.encoder.layer[i_layer].parameters():
                    param.requires_grad = False


def runSentimentModel(dataset, action, data_dir, output_dir, learning_rate, batch_size, n_epochs, is_bert_lstm, lstm_sequence_length,lstm_hidden_layers, lstm_stacked_layers, fast_break, model_type=None, tokenizer=None, debug_gpu_memory=False, optimizer=None, dropout_perc=None, number_layers_freeze=12,local_model_location=None,dense_model_type=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    lstm_args = Object()
    lstm_args.is_bert_lstm = is_bert_lstm
    lstm_args.lstm_sequence_length = lstm_sequence_length
    lstm_args.hidden_layers = lstm_hidden_layers 
    lstm_args.stacked_lstm_layers = lstm_stacked_layers 
    
    global_step = 0
    print('Using ' + device.type, device_name) 
    num_labels = len(dataset.train.labels2idx)
    print('Having ', num_labels, 'classes.')
    print('Learning rate ', learning_rate)

    if action == 'train':
        is_train = True 
    else:
        is_train = False
        last_epoch = -1
        n_epochs = 1
    

    model, last_epoch = get_model(action=action, data_dir=data_dir, output_dir=output_dir, device=device, num_labels=num_labels, lstm_args=lstm_args, model_type=model_type, tokenizer=tokenizer, dropout_perc=dropout_perc,local_model_location=local_model_location,dense_model_type=dense_model_type)
    
    if optimizer is None or "Adam" in optimizer:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif optimizer == "RMSprop":
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    #if action =='train':
    #    drawLabelsDistributionPerStep(dataset.all_label_ids_no_shuffled, batch_size, output_dir, 'class_dist_steps_no_shuffled' + action)
    #drawLabelsDistributionPerStep(dataset.all_label_ids, batch_size, output_dir,'class_dist_steps_shuffled_'+ action)
    
    if not is_train:
        dataset.val = dataset.test 

        counter_classes = Counter(dataset.test.all_label_ids)
        number_classes = len(counter_classes)
        if number_classes > 0: 
            avg_classes = sum([counter_classes[item] for item in counter_classes])/number_classes
            if is_bert_lstm: 
                stdev_classes = statistics.stdev([counter_classes[item]/batch_size for item in counter_classes])
                avg_classes = avg_classes / batch_size
            else:
                stdev_classes = statistics.stdev([counter_classes[item] for item in counter_classes])

        print('Number of classes for validation/test', number_classes,'Average validation/test records per class', int(avg_classes), ' stdev ', stdev_classes)

    else:
        avg_classes = 0
        counter_classes = Counter(dataset.train.all_label_ids)
        if len(counter_classes)>1:
            number_classes = len(counter_classes)
            stdev_classes = None
            if number_classes > 0: 
                avg_classes = sum([counter_classes[item] for item in counter_classes])/number_classes
                if is_bert_lstm:
                    stdev_classes = statistics.stdev([counter_classes[item]/batch_size for item in counter_classes])
                    avg_classes = avg_classes / batch_size
                else:
                    stdev_classes = statistics.stdev([counter_classes[item] for item in counter_classes])

            print('Number of classes for training', number_classes,'Average training records per class', avg_classes, ' stdev ', stdev_classes)
        
    if last_epoch+1 == n_epochs: print('Total number of epochs reached (',n_epochs,")")
    print_memory = torch.cuda.is_available() and debug_gpu_memory 
    loss_historical = {}
    
    for epoch in range(last_epoch+1, n_epochs):
        loss_historical[epoch] = []
        print('New epoch ', epoch) 
        if is_train:
            if number_layers_freeze >0:
                set_no_grad(model, number_layers_freeze, is_train)

            print('Start training...')
            epoch_loss = 0.0
            labels_list = []
            preds_list = []
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            batch_size_steps = lstm_args.lstm_sequence_length*batch_size* lstm_args.stacked_lstm_layers  if lstm_args.is_bert_lstm else batch_size 
            
            model.train()
                        
            print('Batch size ', batch_size_steps)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if print_memory: print('Cache empty before running', torch.cuda.memory_summary())
            
            loop = tqdm.tqdm(range(0, len(dataset.train.all_input_ids), batch_size_steps))
            min_loss = 100
            small_step = 0
            i_nonproc_batch = 0
            current_loss =0
            for i_start in loop:
                loop.set_description(f'Epoch {epoch}') 
                #with torch.set_grad_enabled(True):
                optimizer.zero_grad() #nueva linea
                i_end = i_start + batch_size_steps

                input_ids = dataset.train.all_input_ids[i_start:i_end]
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
                if print_memory: print('Sent input_ids to cuda',  torch.cuda.memory_summary())
                attention_mask = dataset.train.all_input_mask[i_start:i_end]
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
                if print_memory: print('Sent attention_mask to cuda', torch.cuda.memory_summary())
                token_type_ids = dataset.train.all_segment_ids[i_start:i_end]
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device)
                if print_memory: print('Sent token_type_ids to cuda', torch.cuda.memory_summary())
                labels = dataset.train.all_label_ids[small_step:small_step+batch_size]
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                small_step += batch_size
                
                if print_memory: print('Sent labels to cuda', torch.cuda.memory_summary())

                try:       
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, weights=dataset.train.weights, lstm_hidden=None) #return_dict
                except Exception as ex:
                    print('Batch not processed',i_nonproc_batch)
                    i_nonproc_batch +=1
                    print(ex)
                    gc.collect()
                    raise ex
                if print_memory: print('Forward pass completed', torch.cuda.memory_summary())

                #calculate loss.
                current_loss = outputs['loss']
                currrent_loss_f = current_loss.item()
                epoch_loss += currrent_loss_f
                loss_historical[epoch].append(currrent_loss_f)

                loop.set_postfix(loss=currrent_loss_f)
                #calculate accuracy and keep  for metrics
                pred_label_batch = outputs['logits'].argmax(-1).detach().cpu().numpy()
                y_label_batch = labels.detach().cpu().numpy()

                if not (currrent_loss_f >1 and min_loss < 3): 
                #print(i_start,'Loss increase', pred_label_batch, 'should be: ',labels,'Loss:', currrent_loss_f)
                #else:
                    min_loss = currrent_loss_f

                #backward pass
                current_loss.backward()
                if print_memory: print('Backward completed', torch.cuda.memory_summary())

                #update weights
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if print_memory: print('Optimizer updated', torch.cuda.memory_summary())

                input_ids.detach().cpu()
                attention_mask.detach().cpu()
                token_type_ids.detach().cpu()
                if print_memory: print('Detached objects', torch.cuda.memory_summary())
                del input_ids
                del attention_mask
                del token_type_ids
                del outputs

                if print_memory:print('Deleted objects', torch.cuda.memory_summary())

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if print_memory :print('Cache empty', torch.cuda.memory_summary())


                if is_bert_lstm:
                    preds_list.extend([item for item in pred_label_batch])
                    labels_list.extend([item for item in y_label_batch])
                    #labels_list.extend(y_label_batch.reshape(current_batch_size,lstm_args.lstm_sequence_length)[:,-1])
                else:
                    preds_list.extend([item for item in pred_label_batch])
                    labels_list.extend([item for item in y_label_batch])
                    check_ = {item:0 for item in pred_label_batch}
                    if len(check_)==1 and len({item:0 for item in y_label_batch}) >1:
                        print('Training: All classes were predicted as ',list(check_.keys())[0],' having as labels', y_label_batch)

                if fast_break and i_start > 200: break
            
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    

            print('Finished epoch. Computing accuracy...')
        
            output_dir = os.path.join(data_dir, output_dir)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
    
            model.save_pretrained(output_dir)
            print('Saved pretrained model in ' + output_dir)
    
            write_metrics(labels_list, preds_list, output_dir, action, model_type, global_step, len(dataset.train.all_input_ids), device, n_epochs, epoch, current_loss=current_loss, weights = dataset.train.weights, split_type=action)

            print('Confusion Matrix:')
            print(dataset.train.labels2idx)
            print(confusion_matrix(labels_list, preds_list, labels=np.arange(len(dataset.train.labels2idx))))

            draw_loss(output_dir, loss_historical)

        #VALIDATION / TEST #data is not feed correctly
        model.eval()
        preds_list_val = []
        labels_list_val = []
        ypred_logits = []
        ytruth_logits = []
        
        batch_size_steps = lstm_args.lstm_sequence_length*batch_size*lstm_args.stacked_lstm_layers if lstm_args.is_bert_lstm else batch_size
        small_step = 0
        i_nonproc_batch = 0
        for i_start in tqdm.tqdm(range(0, len(dataset.val.all_input_ids), batch_size_steps)):
            #torch.cuda.clear_memory_allocated() #entirely clear all allocated memory

            with torch.set_grad_enabled(False):
                i_end = i_start + batch_size_steps
                
                input_ids = dataset.val.all_input_ids[i_start:i_end]
                input_ids = torch.tensor(input_ids, dtype=torch.long).to(device) 
                attention_mask = dataset.val.all_input_mask[i_start:i_end]
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device) 
                token_type_ids = dataset.val.all_segment_ids[i_start:i_end]
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device) 
                labels = dataset.val.all_label_ids[small_step:small_step + batch_size]
                labels = torch.tensor(labels, dtype=torch.long).to(device)  
                small_step = small_step + batch_size
                if input_ids.shape[0] ==lstm_args.lstm_sequence_length*lstm_args.stacked_lstm_layers :
                    print('')
                #if is_bert_lstm:
                #    labels = labels.reshape(-1,lstm_args.lstm_sequence_length)[:,0]
                #current_batch_size = len(labels)

                #forward pass 
                try:       
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels) 
                except Exception as ex:
                    print('Batch not processed',i_nonproc_batch)
                    i_nonproc_batch +=1
                    print(ex)
                    gc.collect()
                    raise ex

                
                #calculate accuracy and keep  for metrics 
                pred_label_batch = outputs['logits'].detach().cpu().numpy()
                if is_bert_lstm:
                    ypred_logits.extend(list(pred_label_batch))
                else:
                    ypred_logits.extend(list(pred_label_batch))

                pred_label_batch = pred_label_batch.argmax(-1)

                y_label_batch = labels.detach().cpu().numpy()
                #if is_bert_lstm:
                #    ytruth_logits.extend(y_label_batch.reshape(current_batch_size,lstm_args.lstm_sequence_length)[:,-1])
                #else:            
                 #   ytruth_logits.extend(list(y_label_batch))
                ytruth_logits.extend(list(y_label_batch))
                outputs = None
                del outputs
                
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if print_memory :print('Cache empty', torch.cuda.memory_summary())
                    
                if is_bert_lstm:
                    preds_list_val.extend([item for item in pred_label_batch])
                    labels_list_val.extend(y_label_batch)#.reshape(current_batch_size,lstm_args.lstm_sequence_length)[:,-1])
                else:
                    preds_list_val.extend([item for item in pred_label_batch])
                    labels_list_val.extend([item for item in y_label_batch])
                    
                    check_ = {item:0 for item in preds_list_val}
                    if len(check_)==1 and len({item:0 for item in y_label_batch}) >1:
                        print('Validation/Test: All classes were predicted as ',list(check_.keys())[0],' having as labels', y_label_batch)
    

    
        split_type = 'val' if action =='train' else 'test'
        print('Confusion Matrix:')
        print(dataset.train.labels2idx)
        print(confusion_matrix(labels_list_val, preds_list_val, labels=np.arange(len(dataset.train.labels2idx))))
        write_metrics(labels_list_val, preds_list_val, output_dir, action, model_type, 0, len(dataset.val.all_input_ids), device, n_epochs, epoch, current_loss=None, weights = dataset.train.weights, split_type=split_type)
        write_logits(ytruth_logits, ypred_logits, number_classes, dataset.train.weights, output_dir)

def draw_auc(ytruth, ypred, number_classes, weights):
    y_pred_prob = []
    weights_list = []
    for y_pred_item in ypred:
        if len(y_pred_item) == number_classes:
            temp_vect = np.array([int(item*1000)/1000 for item in y_pred_item]) 
            y_pred_prob.append((temp_vect-min(temp_vect))/sum(temp_vect-min(temp_vect)))
            weights_list.append(weights[np.argmax(y_pred_item)])
        else:
            print('')
    return roc_auc_score(y_true= ytruth,y_score = y_pred_prob,sample_weight = weights_list, multi_class='ovr')

def runROCAnalysis(models_directory, output_dir, title):
    auc_score_list = {}
    for directory in os.walk(models_directory): 
        for result_file in directory[2]:
            oFile= Path(result_file)
            if oFile.suffix=='.pickle':
                with open(os.path.join(directory[0],result_file), 'rb') as f:
                    results = pickle.load(f)
                    auc_score = draw_auc(results['y_truth'], results['y_pred'], results['number_classes'],results['weights'])
                    auc_score_list[oFile.stem] = auc_score
    
    auc_score_list = sorted(auc_score_list.items(), key=lambda x:x[1])
    models = [item[0] for item in auc_score_list]
    values = [item[1] for item in auc_score_list]

    fig, ax = plt.subplots(figsize =(16, 9))
    
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_tick_params(pad = .1)
    ax.yaxis.set_tick_params(pad = .10)
    plt.xlim(min(values)-0.2, 1.1)
    ax.grid(b = True, color ='blue', linestyle ='dashdot', linewidth = 0.5, alpha = 0.2)
    
    bars = ax.barh(models, values) 
    bars[-1].set_color('green')
    bars[0].set_color('red')

    for i in ax.patches:
        plt.text(i.get_width()+0.02, i.get_y()+i._height/2, str(round((i.get_width()), 2)), fontsize = 10, fontweight ='bold', color ='grey')
    ax.set_title('AUC comparison for ' + title, loc ='center', ) 
    
    mean_value = np.nanmean(values)
    plt.axvline(x=mean_value, color = 'blue', linestyle="dotted")
    plt.text(mean_value+0.001, (ax._viewLim.ymin+ax.patches[0].get_y())/2, 'mean='+str(int(mean_value*100)/100), fontsize=10, fontweight ='bold', color ='blue')
    plt.savefig(os.path.join(output_dir,'auc_comparison.jpg'))

def draw_loss(output_dir, loss_historical):
    plt.clf()
    plt.title('Training Loss')
    x_axis = None
    for serie in loss_historical:
        if x_axis is None: x_axis =np.arange(len(loss_historical[serie]))
        y_axis = loss_historical[serie]
        plt.plot(x_axis, y_axis, label='epoch '+ str(serie))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.autoscale(enable=True, axis='both')
    plt.legend()
    plt.savefig(os.path.join(output_dir,'training_loss.jpg'))
        
        
def write_metrics(labels_list, preds_list, output_dir, action, model_type, global_steps, dataset_length, device, n_epochs, current_epoch, current_loss=None, weights = None, split_type=None):
    _accuracy = accuracy_score(labels_list, preds_list)
    _precision = precision_score(labels_list, preds_list, average='weighted')
    _recall =  recall_score(labels_list, preds_list, average='weighted')
    _f1 = f1_score(labels_list, preds_list, average='weighted')
    _loss = float(current_loss.data) if current_loss is not None else 0
    print('Epoch:' + str(current_epoch) + "/" + str(n_epochs-1) + ":")
    print(split_type + " accuracy: ", str(_accuracy))
    print(split_type + " F1: ", str(_f1))
    filename, results = read_metrics(output_dir, model_type, action)
    
    with open(filename, 'wb') as f:
        current_metrics =    {"type": split_type,
                            "precision": _precision,
                            "recall":_recall,
                            "f1": _f1,
                            'loss': _loss,
                            'accuracy': _accuracy,
                            'global_steps': global_steps,
                            'dataset_length': dataset_length, 
                            'device': device.type,
                            'n_epochs': n_epochs,
                            'last_epoch': current_epoch,
                            'weights': [float(item) for item in weights]}
        results.append(current_metrics)

        f.write(json.dumps(results).encode("utf-8")) 
        #f.write(',\n'.join(str(result_metrics).split(',')))
        print("Saved:", f.name)

   
def write_logits(y_truth, preds_list, number_classes, weights, output_directory):   
    filename = os.path.join(output_directory,Path(output_directory).stem + ".pickle")
    current_metrics = {"y_pred": list(preds_list), 
                        "y_truth": list(y_truth),
                        "number_classes":number_classes,
                        "weights":weights}
    
    with open(filename, 'wb') as f:
        pickle.dump(current_metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved:", filename)


def read_metrics(output_dir, model_type, action):
    filename = os.path.join(output_dir, model_type + '_' + action + "_results.json")
    results = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results =  json.load(f) 

    return filename, results
