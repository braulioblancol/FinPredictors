import torch
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients, visualization, LayerGradientXActivation
import webbrowser
import seaborn as sns
import matplotlib as mpl
from sentimentModel import CustomBERTModel
from nlpde import FDExt
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
from wordcloud import WordCloud
import json

class CustomBERTExplainer: 
    def __init__(self, model=None, tokenizer=None, type_model='custom', is_train=False, number_layers_freeze=0): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        if type(model) == str and type_model=='custom':
            self.model = CustomBERTModel(tokenizer=tokenizer, local_model_location=model, transfer_weights=False).from_pretrained(model_dir=model) 
        else:
            self.model = model
        if type(tokenizer)==str and type_model=='custom':
            self.tokenizer =  BertTokenizer.from_pretrained(tokenizer) 
        else:
            self.tokenizer = tokenizer
      
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.type_model = type_model


        requires_grad = True if is_train else False
        for param in self.model.bert.parameters():
            param.requires_grad = requires_grad
        
        if number_layers_freeze >0 and is_train:
            if number_layers_freeze > 12:
                print('Freezing all BERT layers.')
                for param in self.model.bert.parameters():
                    param.requires_grad = False
            else:
                print('Freezing the first:', number_layers_freeze, 'layers of BERT.')
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = False
                for i_layer in range(min(number_layers_freeze, 12)):
                    for param in self.model.bert.encoder.layer[i_layer].parameters():
                        param.requires_grad = False



    def predict(self, inputs, token_type_ids=None, attention_mask=None):
        if self.type_model == 'custom':
            output = self.model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, return_logits=True)
        else:
            output = self.model(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    
        return output

    def explainable_pos_forward_func(self, inputs, token_type_ids=None, attention_mask=None):
        pred = self.predict(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        return pred.max(1).values

    def explainable_pos_forward_func_layers(self, inputs_embeds, token_type_ids=None, attention_mask=None):
        pred = self.model(inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask,return_logits=True)
        
        return pred.max(1).values

    def summarize_attributions(self, attributions, input_tokens, sequence_length, stacked_layers, sequence_shift): #TODO: return also the unified tokens of each group
        tokens_per_group = []
        attributions_per_group = []   
        if sequence_length * stacked_layers >1:
            total_shift = len(input_tokens[0]) - 1 - sequence_shift
            for group_id in range(stacked_layers):
                current_tokens = [] 
                concat_attrib =  torch.tensor([])
                last_segment = []
                for sequence_id in range(sequence_length):
                    record_id = group_id*sequence_length+sequence_id 
                    if input_tokens[record_id][-1] == '[PAD]': #sentece finished before end of the sequence 
                        if len(last_segment)>0:
                            current_segment = attributions[record_id][1:len(last_segment)+1].sum(dim=-1).squeeze(0)
                            current_segment = (current_segment + last_segment)/2 #average the attributions of the shifted part
                            concat_attrib =  torch.cat((concat_attrib, current_segment), dim=0)
                            concat_attrib =  torch.cat((concat_attrib, attributions[record_id][total_shift:-1].sum(dim=-1).squeeze(0)), dim=0)
                        else:
                            concat_attrib  =  torch.cat((concat_attrib, attributions[record_id][1:-1].sum(dim=-1).squeeze(0)), dim=0)

                        sep_index = [i for i, item in enumerate(input_tokens[record_id]) if item =='[SEP]']
                        if len(sep_index)>0:
                            sep_index=sep_index[0]
                            current_tokens.extend(input_tokens[record_id][len(last_segment)+1:sep_index])
                        break #dont continue the followings
                    else: #ends with 'SEP' token
                        concat_attrib  =  torch.cat((concat_attrib,attributions[record_id][1:-total_shift].sum(dim=-1).squeeze(0)), dim=0)
                        last_segment = attributions[record_id][-total_shift:-1].sum(dim=-1).squeeze(0)
                        current_tokens.extend(input_tokens[record_id][1:-1])
                concat_attrib = concat_attrib[:len(current_tokens)]
                attributions_per_group.append(concat_attrib / torch.norm(concat_attrib))
                tokens_per_group.append(current_tokens)
        else:
            record_id =0
            sep_index = [i for i, item in enumerate(input_tokens[record_id]) if item =='[SEP]'][0]
            tokens_per_group.append(input_tokens[record_id][1:sep_index]) 
            attributions_group = attributions[record_id].sum(dim=-1).squeeze(0)[1:sep_index]  
            attributions_per_group.append(attributions_group /  torch.norm(attributions_group)) # norm = sqrt(sum(square(x)))

        return attributions_per_group, tokens_per_group
    
    def explain_model_input(self, data_dir, output_dir, max_samples, total_steps, dataset_name, max_significant_words=2, tokenizer=None):
        oDataset = FDExt(data_dir, output_dir, dataset_name=dataset_name)
        oDataset.loadDataset() 
        #args_ret = oDataset.prepare_input_data(model_dir= model_dir, model_type = model_type, group_level=None, y_label_type=y_label_type, labels=None, y_top_n=y_top_n, trim_begining=False,tokenizer=tokenizer,sequence_strategy=sequence_strategy, sequence_shift=sequence_shift, max_segments_bert=max_segments_bert, terms_filter_path=terms_list, clean_data=clean_data, remove_numbers=remove_numbers,time_grouped_by_company=time_grouped_by_company)
        self.tokenizer = oDataset.args_ret['tokenizer']  if tokenizer is None else BertTokenizer.from_pretrained(tokenizer)
        ig = LayerIntegratedGradients(self.explainable_pos_forward_func, self.model.bert.embeddings)
        
        oDataset.val.idx2labels = oDataset.train.idx2labels
        oDataset.train = None
        gc.collect()

        total_labels = 0
        #vis_data_records = []    
        class_results = {}
        top_significant_words = {}
        bottom_significant_words = {}
        recordset_size = oDataset.args_ret['lstm_stacked_layers']*oDataset.args_ret['lstm_sequence_length']
        for i_input  in range(0,len(oDataset.val.text),recordset_size):
            if recordset_size ==1:
                text = oDataset.val.text[i_input].replace("'"," ")
                encoded_text =  self.tokenizer.encode_plus(text) 
                tokenized_text = self.tokenizer.tokenize(text)
                if len(tokenized_text) >510:
                    continue
                input_tokens = [['[CLS]'] + tokenized_text + ['[SEP]']]
                masks=  torch.tensor([encoded_text.data['attention_mask']]).to(self.device).long()
                token_types= torch.tensor([encoded_text.data['token_type_ids']]).to(self.device).long()
            else:
                input_tokens = oDataset.val.all_tokens[i_input:i_input+recordset_size]
                masks=  torch.tensor(oDataset.val.all_input_mask[i_input:i_input+recordset_size]).to(self.device).long()
                token_types= torch.tensor(oDataset.val.all_segment_ids[i_input:i_input+recordset_size]).to(self.device).long()
            label =oDataset.val.all_label_ids[i_input:i_input+oDataset.args_ret['lstm_stacked_layers']]
            gt_label = oDataset.val.idx2labels[label[0]]
            if class_results.get(gt_label) is None or len(class_results[gt_label]) < max_samples_per_label:
                if recordset_size ==1:
                    input_ids = [encoded_text.data['input_ids']]
                else:
                    input_ids = oDataset.val.all_input_ids[i_input:i_input+recordset_size]
                input_ids = torch.tensor(input_ids).to(self.device)
                input_ids = input_ids.long()

                #inputs_embeds = model.bert.embeddings.word_embeddings(torch.tensor(oDataset.all_input_ids[i_input]).to(device)).detach().numpy()
                input_baseline = torch.zeros_like(input_ids) #self.tokenizer.encode(' '.join([item if item in self.tokenizer.all_special_tokens else self.tokenizer.pad_token  for item in input_id[1:-1]]))
                input_baseline = input_baseline.to(self.device).long()
                #input_baseline = zip(input_baseline,oDataset.all_input_mask[i_input],oDataset.all_segment_ids[i_input])

                output_pred = self.model(input_ids, attention_mask=masks, token_type_ids=token_types, return_logits=True)
                pred = output_pred.argmax().item()
                predicted_label = oDataset.val.idx2labels[pred]
                max_samples_per_label = max(1,int(max_samples/len(oDataset.val.idx2labels)))


                if pred == label[0]:
                    attributions, delta = ig.attribute(inputs=(input_ids, token_types),
                                        baselines=(input_baseline, token_types),
                                        additional_forward_args=(masks), 
                                        n_steps = total_steps,
                                        return_convergence_delta=True)
                    
                    attributions_sum, tokens_sequence = self.summarize_attributions(attributions, input_tokens, oDataset.args_ret['lstm_sequence_length'], oDataset.args_ret['lstm_stacked_layers'], oDataset.args_ret['sequence_shift'])
                    #groups_length = [len(item) for item in tokens_sequence]
                    #number_tokens = len([item for item in input_tokens if item != self.tokenizer.pad_token])
                    #attributions_text = attributions_sum[:number_tokens-1]
            
                    attributions_text = torch.tensor([])
                    tokens_text = []
                    for i_record, attribution_record in enumerate(attributions_sum):
                        attributions_text = torch.cat((attributions_text,attribution_record), dim=0)
                        tokens_text.extend(tokens_sequence[i_record])
                        if i_record < len(attributions_sum)-1:    
                            tokens_text[-1] += "\n\n\n\n"

                    predicted_score_norm = ((output_pred+-output_pred.min().item())/ (output_pred+-output_pred.min().item()).sum(dim=-1).squeeze(0)).max().item()

                    results_vis = visualization.VisualizationDataRecord(
                                    attributions_text,
                                    predicted_score_norm, #output_pred.max().item(), #predicted score
                                    predicted_label, #predicted
                                    oDataset.val.idx2labels[label[0]], #true
                                    predicted_label + '(' + str(total_steps) + ')', #attribution label (ground truth)
                                    attributions_text.sum(), #attribution score
                                    tokens_text,
                                    delta)
                                    
                    if class_results.get(predicted_label) is None: class_results[predicted_label] = []
                    class_results[predicted_label].append(results_vis) 
                    total_labels += 1
                    
                    vect_attributions = [(item.item(),i) for i,item in enumerate(attributions_sum[0])]    
                    vect_attributions.sort(reverse=True)
                    top_list_indexes = [item[1] for item in vect_attributions[:max_significant_words]]
                    
                    top_list = [tokens_sequence[0][index] for index in top_list_indexes]
                    if top_significant_words.get(predicted_label) is None: top_significant_words[predicted_label]=[]
                    top_significant_words[predicted_label].extend(top_list)

                    bottom_list_indexes = [item[1] for item in vect_attributions[-max_significant_words:]]
                    bottom_list = [tokens_sequence[0][index] for index in bottom_list_indexes]
                    if bottom_significant_words.get(predicted_label) is None: bottom_significant_words[predicted_label]=[]
                    bottom_significant_words[predicted_label].extend(bottom_list)
            if total_labels >= max_samples:
                break

        vis_explanation = visualization.visualize_text([results_item for item in class_results for results_item in class_results[item] ])
        
        xp_file = os.path.join(output_dir,'attributions_list.html')
        with open(xp_file, 'w') as f:
            f.write(vis_explanation.data)
            f.close()
        
        webbrowser.open(xp_file) 
        
        top_bottom_text_list = os.path.join(output_dir,'word_cloud_data.json')
        with open(top_bottom_text_list,'w') as f:
            data = {'top':top_significant_words, 'bottom':bottom_significant_words}
            json.dump(data, f) 

        for type_significance in ['top','bottom']:
            for label in data[type_significance]:
                self.draw_word_cloud(data[type_significance][label],output_dir,type_significance,label) 

    def draw_word_cloud(self, word_list, output_dir, type_significance, class_name):
        wordcloud_list = WordCloud(width = 800, height = 800, background_color ='white', min_font_size = 10).generate(' '.join(word_list))
        plt.figure(figsize = (15, 15), facecolor = None)
        if type_significance=='top':
            plt.title('Most significative words for predicting "' + class_name.upper()+ '"')
        else:
            plt.title('Less significative words for predicting "' + class_name.upper() + '"')
        plt.imshow(wordcloud_list)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        file_path = os.path.join(output_dir,'wordcloud_' + type_significance + "_" + class_name + '.png')
        plt.savefig(file_path)

    def explain_model_layers(self, data_dir, model_dir, output_dir, filter_last_doc,labels_path,perc_data,filter_lang,model_type,y_label_type,y_top_n,tokenizer,sequence_strategy,remove_stopwords,sequence_shift,max_segments_bert,terms_list,clean_data,remove_numbers,time_grouped_by_company, max_samples, total_steps, dataset_name=None):
        oDataset = FDExt(data_dir, output_dir,dataset_name=dataset_name)
        oDataset.loadDataset(filter_last_doc=filter_last_doc, filter_type_doc='eCDF', additional_filters={'page_type':['Unknown','']},perc_data=perc_data, labels_path=labels_path, data_sel_position='last', filter_lang=filter_lang) 
        args_ret = oDataset.prepare_input_data(action='test', model_dir= model_dir, model_type = model_type, group_level=None, y_label_type=y_label_type,y_top_n=y_top_n, trim_type="end",tokenizer=tokenizer,sequence_strategy=sequence_strategy, remove_stopwords= remove_stopwords, sequence_shift=sequence_shift, max_segments_bert=max_segments_bert, terms_filter_path=terms_list, clean_data=clean_data, remove_numbers=remove_numbers,time_grouped_by_company=time_grouped_by_company)
        self.tokenizer = args_ret['tokenizer'] 
        
        #layer_attrs_dist = []
        layer_attrs = []    
        ig = LayerIntegratedGradients(self.explainable_pos_forward_func, self.model.bert.embeddings)
        
        for i_input, input_id in enumerate(oDataset.all_tokens):
                masks=  torch.tensor([oDataset.all_input_mask[i_input]]).to(self.device).long()
                token_types= torch.tensor([oDataset.all_segment_ids[i_input]]).to(self.device).long()
                label =oDataset.all_label_ids[i_input]
                input_baseline = self.tokenizer.encode(' '.join([item if item in self.tokenizer.all_special_tokens else self.tokenizer.pad_token  for item in input_id[1:-1]]))
                input_baseline = torch.tensor([input_baseline]).to(self.device).long()
                
                input_ids = oDataset.all_input_ids[i_input]
                input_ids = torch.tensor([input_ids]).to(self.device)
                input_ids = input_ids.long()

                output_pred = self.model(input_ids, attention_mask=masks, token_type_ids=token_types, return_logits=True)
                pred = output_pred.argmax().item()

                if pred == label and oDataset.idx2labels[pred] !='General':
                    if oDataset.all_words[i_input][:37] == 'les droits audiovisuels correspondent' and oDataset.all_metadata[i_input]['document'] == '9lHt7P' and oDataset.all_metadata[i_input]['page'] == 10 and oDataset.idx2labels[pred]  == 'Assets':                
                        number_tokens = len([item for item in input_id if item != self.tokenizer.pad_token])
            
                        attributions_input, delta = ig.attribute(inputs=(input_ids, token_types), baselines=(input_baseline, token_types),
                                                            additional_forward_args=(masks), n_steps = total_steps, return_convergence_delta=True)

                        attributions_input_sum = self.summarize_attributions(attributions_input)[:number_tokens-1].cpu().detach().tolist()
                        
                        all_tokens = oDataset.all_tokens[i_input][:number_tokens-1]
                        topK_index = np.argsort(attributions_input_sum[:number_tokens-1])[:5]
                        topk_tokens = [item for i,item in enumerate(all_tokens) if i in topK_index]

                        for i in range(self.model.bert.config.num_hidden_layers):
                            lc = LayerGradientXActivation(self.explainable_pos_forward_func_layers, self.model.bert.encoder.layer[i])
                            layer_attributions = lc.attribute(inputs=(input_ids,token_types),  additional_forward_args=(masks)) #baselines=(input_baseline,token_types),
            
                            attributions_sum = self.summarize_attributions(layer_attributions).cpu().detach().tolist()
                            layer_attrs.append(attributions_sum[:number_tokens-1])
                            #layer_attrs_dist.append(layer_attrs[0,token_to_explain,:].cpu().detach().tolist())
                    

                        break     
                 
                 
        
        #fig, axs = plt.subplots(10,1,figsize=(15,5), sharey=True)
        

        xticklabels=all_tokens
        yticklabels=list(range(1,13))
        cmap = mpl.cm.YlOrRd_r
        
        fig = plt.figure(constrained_layout=True)  
        ax = fig.add_gridspec(10, 1)
        ax1 = fig.add_subplot(ax[0:9, 0])
        sns.heatmap(np.array(layer_attrs), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2, cmap=cmap,ax=ax1)
        plt.xlabel('Tokens')
        plt.ylabel('Layers')

        # Create the second figure
        ax2 = fig.add_subplot(ax[9, 0])
        sns.heatmap([attributions_input_sum], annot=False, cmap=cmap)

        for i, (lab, annot) in enumerate(zip(ax2.get_yticklabels(), ax2.texts)):
            text =  lab.get_text()
            if i in topk_tokens: # rows to highlight 
                lab.set_weight('bold')
                lab.set_size(20)
                #lab.set_color('purple')

                annot.set_weight('bold')
                #annot.set_color('purple')
                annot.set_size(20)

        plt.show()
        print('')
