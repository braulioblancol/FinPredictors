from cv2 import split
import tensorflow as tf
import tqdm
import ast
import time
import logging
import argparse
import os
import gc
import torch
import torch.nn as nn
from torch import optim
from transformers import BertModel, BertTokenizer, BertConfig 
from nlpde import FDExt  
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
'''
class CustomBERTModelConfig(BertConfig):
    model_type = "sentimentBERT"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        max_2d_position_embeddings=1024,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )

'''
 
class Object(object):
    pass

class CustomBERTModel(nn.Module):
    def __init__(self, number_labels = 2, device=None, model_type='SA', dropout_perc = 0.1, is_bert_lstm=True, hidden_layers=10,stacked_lstm_layers=5, tokenizer=None):
        super(CustomBERTModel, self).__init__()
        self.args = {'number_labels':number_labels,'device':device,'dropout_perc':dropout_perc, 'model_type': model_type}
        self.model_type = model_type
        self.num_labels = number_labels
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased", output_hidden_states = True,)
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        #batch_sentences = ["But what about second breakfast?", "Don't think he knows about second breakfast, Pip.", "What about elevensies?", ]
        #inputs = self.tok(batch_sentences, return_tensors="pt")
        
        #if tokenizer is not None:
        #    self.bert.embeddings = nn.Embedding(len(tokenizer), self.bert.embeddings.word_embeddings.embedding_dim)

        self.bert.to(device)
        self.dropout_perc = dropout_perc
        self.dropout = nn.Dropout(self.dropout_perc)
        self.is_bert_lstm = is_bert_lstm
        if self.is_bert_lstm:
            self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_layers, stacked_lstm_layers, bias=True, batch_first=True, dropout=self.dropout_perc)
            self.classifier = nn.Linear(hidden_layers, self.num_labels)
        else:
            self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        #self.config_class = SentimentBERTForClassificationConfig
        #self.config = self.config_class.from_pretrained(self.model_type, num_labels=self.num_labels, cache_dir=None)

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            weights=None
    ):
        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict
        
        #embeds = self.bert(input_ids=input_ids, attention_mask= attention_mask, labels=labels)

        embeds = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = embeds[1] #[0]:entire output, token by token ,[1]: output only for the first token [CLS]
        
        hidden_states = embeds.hidden_states
        attentions = embeds.attentions
        
        if return_dict:
            del embeds

        sequence_output = self.dropout(sequence_output)
        if self.is_bert_lstm:
            sequence_output = torch.unsqueeze(sequence_output, 0)
            sequence_output, _ = self.lstm(sequence_output) #TODO: Improve the initial state self.num_bert_batchs 
            logits = self.classifier(sequence_output)
        else:            
            logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if weights is not None:
                class_weights = torch.FloatTensor(weights).to(input_ids.device.type)
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # 10,5120

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
                'hidden_states':hidden_states,
                'attentions':attentions,
            }

    def from_pretrained(self, model_dir):
        args = torch.load(os.path.join(model_dir, "training_args.bin"))
        print('Loading from pre-trained model ', args)
        model_file = os.path.join(model_dir, "pytorch_model.bin")
        model = CustomBERTModel(**args)
        model.load_state_dict(torch.load(model_file))
        return model

    def save_pretrained(self, output_dir):
        assert os.path.isdir(output_dir)
        self.args['is_bert_lstm'] = self.is_bert_lstm

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # Save configuration file
        #model_to_save.save_pretrained(output_dir)
        logging.info("Model weights saved in {}".format(output_model_file))

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

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default='train',
        type=str,
        required=True, )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True, )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True, )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        required=False, )

    parser.add_argument(
        "--is_bert_lstm",
        default=True,
        type=bool,
        required=False, )

    parser.add_argument(
        "--sequence_length",
        default=512,
        type=int,
        required=False, )

    parser.add_argument(
        "--batch_size",
        default=10,
        type=int,
        required=False, )

    parser.add_argument(
        "--n_epochs",
        default=1,
        type=int,
        required=False, )

    parser.add_argument(
        "--dataset_perc",
        default=0.01,
        type=float,
        required=False, )

    parser.add_argument(
        "--sequence_strategy",
        default="document_batch",
        type=str,
        required=False, )

    parser.add_argument(
        "--remove_stopwords",
        default=True,
        type=bool,
        required=False, )

    parser.add_argument(
        "--sequence_shift",
        default=200,
        type=int,
        required=False, )

        
    parser.add_argument(
        "--lstm_hidden_layers",
        default=2,
        type=int,
        required=False, )

    parser.add_argument(
        "--fast_break",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--model_type",
        default="SA",
        type=str,
        required=False, )
        
    parser.add_argument(
        "--y_top_n",
        default=50,
        type=int,
        required=False, )     
    
    args = parser.parse_args()    

    oDataset = FDExt(args.data_dir, args.output_dir)    
    oDataset.loadDataset(filter_last_doc=True, filter_type_doc='eCDF', perc_data=args.dataset_perc, additional_filters={'page_type':'Unknown'}) 
    args_ret = oDataset.prepare_input_data(max_sequence_length=args.sequence_length, model_type =args.model_type,group_level=None, y_label_type=None, y_top_n=None, trim_begining=False,tokenizer="bert-base-multilingual-uncased",sequence_strategy=args.sequence_strategy, remove_stopwords=args.remove_stopwords, sequence_shift=args.sequence_shift, max_segments_bert=None)
    oDataset.splitDataset(shuffle = True, test_size= 0.25, trainable_size=0.5, train_size=0.70)
    runSentimentModel(dataset=oDataset,action=args.action,data_dir=args.data_dir,output_dir=args.output_dir,learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,batch_size=args.batch_size, n_epochs=args.n_epochs, is_bert_lstm=args_ret['is_bert_lstm'], 
                lstm_sequence_length=args_ret['lstm_sequence_length'], lstm_hidden_layers = args.lstm_hidden_layers, lstm_stacked_layers=args_ret['lstm_staked_layers'],
                fast_break=args.fast_break, model_type = args.model_type, tokenizer = args_ret['tokenizer'])

def get_model(action, data_dir, output_dir, num_labels, device, n_epochs,lstm_args, model_type=None, tokenizer=None):
    model_dir = os.path.join(data_dir, output_dir)
    model = None
    last_epoch = -1
    if action == 'train':
        for file in os.listdir(model_dir):
            if 'pytorch_model.bin' == file:       
                model = CustomBERTModel(device=device,number_labels=num_labels,is_bert_lstm=lstm_args.is_bert_lstm, hidden_layers=lstm_args.hidden_layers,stacked_lstm_layers=lstm_args.stacked_lstm_layers, tokenizer = tokenizer).from_pretrained(model_dir=model_dir)
                with open(os.path.join(model_dir, action + "_results.txt"), 'r+', encoding='utf-8') as f:
                    result_metrics = ast.literal_eval(f.read())
                    if result_metrics.get('last_epoch') is not None: last_epoch = result_metrics['last_epoch']
                break
        n_epochs = n_epochs - (last_epoch + 1)
        if model is None:
            model = CustomBERTModel(device=device, number_labels=num_labels,is_bert_lstm= lstm_args.is_bert_lstm,hidden_layers= lstm_args.hidden_layers,stacked_lstm_layers=lstm_args.stacked_lstm_layers, tokenizer=tokenizer)
        model.train()
    elif action == 'test':
        n_epochs = 1
        model = CustomBERTModel().from_pretrained( model_dir=os.path.join(data_dir, output_dir))
        model.eval()
    model.to(device)
    return model, n_epochs


def runSentimentModel(dataset, action, data_dir, output_dir, learning_rate, sequence_length, batch_size, n_epochs, is_bert_lstm, lstm_sequence_length,lstm_hidden_layers, lstm_stacked_layers,fast_break, model_type=None, tokenizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_args = Object()
    lstm_args.is_bert_lstm = is_bert_lstm
    lstm_args.lstm_sequence_length = lstm_sequence_length
    lstm_args.hidden_layers = lstm_hidden_layers 
    lstm_args.stacked_lstm_layers = lstm_stacked_layers 
    
    global_step = 0
    print('Using ' + device.type) 
    num_labels = len(dataset.labels2idx)

    is_train = True if action == 'train' else False

    model, n_epochs = get_model(action=action, data_dir=data_dir, output_dir=output_dir, device=device, num_labels=num_labels, n_epochs=n_epochs, lstm_args=lstm_args, model_type=model_type, tokenizer=tokenizer)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if not is_train:
        dataset.all_input_ids = dataset.test.all_input_ids 
        dataset.all_input_mask = dataset.test.all_input_mask
        dataset.all_segment_ids = dataset.test.all_segment_ids
        dataset.all_label_ids = dataset.test.all_label_ids

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        labels_list = []
        preds_list = []
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = lstm_args.stacked_lstm_layers if lstm_args.is_bert_lstm else batch_size
        #TRAINING OR TESTING
        print('Sequence_lenght ' , sequence_length, ' batch size input ',  batch_size, ' selected batch size ', batch_size)
        for i_start in tqdm.tqdm(range(0, len(dataset.all_input_ids), batch_size)):
            with torch.set_grad_enabled(is_train):
                i_end = i_start + batch_size
                input_ids = dataset.all_input_ids[i_start:i_end].to(device)
                attention_mask = dataset.all_input_mask[i_start:i_end].to(device)
                token_type_ids = dataset.all_segment_ids[i_start:i_end].to(device)
                labels = dataset.all_label_ids[i_start:i_end].to(device)

                #forward pass 
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, weights=dataset.weights) #return_dict
                
                #calculate loss.
                current_loss = outputs['loss']
                epoch_loss += current_loss.item()

                if is_train:
                    #backward pass
                    current_loss.backward()

                    #update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                #calculate accuracy and keep  for metrics
                pred_label_batch = outputs['logits'].argmax(-1).detach().cpu().numpy()
                y_label_batch = labels.detach().cpu().numpy()

                labels_list.extend([item for item in y_label_batch])
                if is_bert_lstm:
                    preds_list.extend([item for item in pred_label_batch[0]])
                else:
                    preds_list.extend([item for item in pred_label_batch])

                if fast_break and i_start > 200: break
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        print('Finished epoch. Computing accuracy...')
       
        output_dir = os.path.join(data_dir, output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if is_train:
            model.save_pretrained(output_dir)
            print('Saved pretrained model in ' + output_dir)
            split_type = 'Train'
        else:
            split_type = 'Test'

        write_metrics(labels_list, preds_list, output_dir, action, global_step, len(dataset.all_input_ids), device, n_epochs, epoch, current_loss=current_loss, weights = dataset.weights, split_type=split_type)
           
        #VALIDATION 
        if is_train:
            preds_list_val = []
            labels_list_val = []
            for i_start in tqdm.tqdm(range(0, len(dataset.all_input_ids), batch_size)):
                with torch.set_grad_enabled(False):
                    i_end = i_start + batch_size
                    input_ids = dataset.val.all_input_ids[i_start:i_end].to(device)
                    attention_mask = dataset.val.all_input_mask[i_start:i_end].to(device)
                    token_type_ids = dataset.val.all_segment_ids[i_start:i_end].to(device)
                    labels = dataset.val.all_label_ids[i_start:i_end].to(device)

                    #forward pass 
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                    
                    #calculate accuracy and keep  for metrics
                    pred_label_batch = outputs['logits'].argmax(-1).detach().cpu().numpy()
                    y_label_batch = labels.detach().cpu().numpy()

                    labels_list_val.extend([item for item in y_label_batch])
                    if is_bert_lstm:
                        preds_list_val.extend([item for item in pred_label_batch[0]])
                    else:
                        preds_list_val.extend([item for item in pred_label_batch])

        
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            write_metrics(labels_list_val, preds_list_val, output_dir, action, 0, len(dataset.val.all_input_ids), device, n_epochs, epoch, current_loss=current_loss, weights = dataset.weights, split_type='Val')
                       

def write_metrics(labels_list, preds_list, output_dir, action,global_steps, dataset_length, device, n_epochs, current_epoch, current_loss=None, weights = None, split_type=None):
    _accuracy = accuracy_score(labels_list, preds_list)
    _precision = precision_score(labels_list, preds_list, average='weighted')
    _recall =  recall_score(labels_list, preds_list, average='weighted')
    _f1 = f1_score(labels_list, preds_list, average='weighted')
    _loss = float(current_loss.data) if current_loss is not None else 0
    print('Epoch:' + str(current_epoch) + "/" + str(n_epochs) + ":")
    print(split_type + " accuracy: ", str(_accuracy))
    print(split_type + " F1: ", str(_f1))
    with open(os.path.join(output_dir, action + "_results.txt"), 'a+', encoding='utf-8') as f:
        result_metrics =    {"type": split_type,
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
                            'type': action,
                            'weights': weights}

        f.write(',\n'.join(str(result_metrics).split(',')))
        print("Saved:", f.name)


if __name__ =='__main__':
    print('Running sentiment model')
    run()