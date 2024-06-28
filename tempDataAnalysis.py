import pickle
from nlpde import FDExt
from collections import Counter
import numpy as np




with open("D:\\datasets\\train_dataset_s3.pickle", 'rb') as f: 
    dataset = pickle.load(f)

for item in dataset:
    print('')
print('')


with open('D:\\datasets\\LBR\\dataset_json\\text_doc.txt','r',encoding='utf-8') as f:
    dataset_docs = f.readlines()

i=0
document_dataset = {}
for line in dataset_docs:
    if i==0:
        columns_indexes = {i:item for i,item in enumerate(line[:-1].split('\t'))}
    else:
        record = {columns_indexes[i]:item for i,item in enumerate(line[:-1].split('\t'))}
        document_dataset[record['file_name']]= record

    i+=1

oDataset = FDExt(None, None, 'train', total_records=10,dataset_name='D:\\datasets\\LBR\\dataset_ub10k_3y_train.pickle')
oDataset.loadDataset(filter_last_doc=-1, 
                    filter_type_doc='eCDF', 
                    perc_data=0, 
                    labels_path=None, 
                    data_sel_position=None, 
                    filter_lang=None,
                    max_number_files=None)


with open('D:\\datasets\\LBR\\nace_first_level.txt','r',encoding='utf-8') as f:
    nace_Codes = f.readlines()

nace_Codes_dict = {item.split('\t')[0].strip():item.split('\t')[1][:-1] for item in nace_Codes}


dataset_doc_list = {item['document']:document_dataset[item['document']] for item in oDataset.train.all_metadata}
industries_list = [nace_Codes_dict.get(dataset_doc_list[item]['industry_code'].split('.')[0]) for item in dataset_doc_list ]

c_industries = Counter(industries_list)
company_list = {item['company_id']:0 for item in oDataset.train.all_metadata}
total_not_completed = 0
cia_dict = {}
for company_id in company_list:
    company_id_list = [item for item in oDataset.train.all_metadata if item['company_id']==company_id]
    if len(company_id_list)!=3:
        print('Not completed ', company_id_list)
        total_not_completed +=1
        cia_dict[company_id] = len(company_id_list)

company_list_val = {item['company_id']:0 for item in oDataset.val.all_metadata}
for company_id in cia_dict:
    company_id_list = [item for item in oDataset.val.all_metadata if item['company_id']==company_id]
    if len(company_id_list)>0:
        cia_dict[company_id] += len(company_id_list)

print(cia_dict)


document_list = {item['document']:{'company_id':item['company_id'],
                                    'year':item['year'], 
                                    'label':oDataset.train.all_labels[i], 
                                    'lang': oDataset.train.text_lang[i], 
                                    'pages': item['page'],
                                    'text': oDataset.train.text[i]} for i,item in enumerate(oDataset.train.all_metadata)}
                                     
document_list.update({item['document']:{'company_id':item['company_id'],
                                    'year':item['year'], 
                                    'label':oDataset.val.all_labels[i], 
                                    'lang': oDataset.val.text_lang[i], 
                                    'pages': item['page'],
                                    'text': oDataset.val.text[i]} for i,item in enumerate(oDataset.val.all_metadata)})

documents_per_lang = [document_list[item]['lang'] for item in document_list]
companies_per_lang = [[document_list[item]['lang'],document_list[item]['pages']] for item in document_list]
pages_per_lang = [[document_list[item]['lang'],document_list[item]['pages']] for item in document_list]
lang_text = [[document_list[item]['lang'],document_list[item]['text'].split('\n')] for item in document_list] 
labels_per_lang =  [[document_list[item]['lang'],document_list[item]['label']] for item in document_list] 
labels = {item[1]:0 for item in labels_per_lang}

langs = {'fra':0,'deu':0,'eng':0}
langs_avg_pages = {'fra':0,'deu':0,'eng':0}
langs_mode_pages = {'fra':0,'deu':0,'eng':0}
langs_labels = {'fra':{},'deu':{},'eng':{}}
for lang in langs:
    langs[lang] = sum([item[1] for item in pages_per_lang if item[0]==lang])
    langs_avg_pages[lang] = np.mean([item[1] for item in pages_per_lang if item[0]==lang])
    langs_mode_pages[lang] = Counter([item[1] for item in pages_per_lang if item[0]==lang]).most_common(1)
    for label in labels:
        langs_labels[lang][label] = sum([1 for item in labels_per_lang if item[0]==lang and item[1]==label])


mean_total = np.mean([item[1] for item in pages_per_lang])

all_tokens = oDataset.train.all_tokens
all_tokens.extend(oDataset.val.all_tokens)

langs_tokens = {'fra':0,'deu':0,'eng':0} 
langs_words = {'fra':0,'deu':0,'eng':0} 
langs_tokens_per_doc = {'fra':0,'deu':0,'eng':0} 
langs_words_per_doc = {'fra':0,'deu':0,'eng':0} 
for lang in langs:
    for i_doc, document in enumerate(lang_text):
        if document[0] == lang:
            total_tokens_per_doc = 0
            total_words_per_doc = 0
            for segment in all_tokens[i_doc*10:(i_doc+1)*10]:
                if segment[0] !='[PAD]':
                    valid_tokens = [single_token for single_token in segment if single_token !='[PAD]']
                    total_tokens_per_doc += len(valid_tokens) - 112
            
            total_tokens_per_doc + 110
            total_words_per_doc += sum([1 for item in document[1] for word in item.split(' ') if len(word.strip())>0])
                        
            langs_tokens[lang] += total_tokens_per_doc 
            langs_words[lang] += total_words_per_doc  
            
    langs_tokens_per_doc[lang] = langs_tokens[lang]/len([item for item in lang_text if item[0]==lang])
    langs_words_per_doc[lang] = langs_words[lang]/len([item for item in lang_text if item[0]==lang])

rank_lang = Counter(documents_per_lang).most_common()

print('') 
        
