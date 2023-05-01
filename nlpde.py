#import logging 
import copy
import time
from datetime import datetime
import urllib.request
import os
import argparse
from regex import E 
import tqdm
import json
import numpy as np
from pathlib import Path
import importlib
import timeit
import requests
import random
import re
import torch
import multiprocessing as mp
import ray
from Levenshtein import distance as lev
from torch.utils.data import Dataset
from collections import Counter 
import math
import pickle

import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path 
import fitz 
import pikepdf
from bs4 import BeautifulSoup
from ray.util.multiprocessing.pool import Pool
from sklearn.utils.class_weight import compute_class_weight
import gc 

from transformers import BertTokenizer,AutoTokenizer

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
PRETRAINED_VOCAB_FILES_MAP = { 
        "fra": "https://huggingface.co/flaubert/flaubert_base_uncased/raw/main/vocab.json",
        "eng": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt", 
        "deu": "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt", 
        "chi": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        "fin": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt", 
    }

langs = {'fra':'fr','eng':'en','deu':'de'}

async_semaphore = None

def getFileFromLink(link, destination_path):
    try:
        response = urllib.request.urlopen(link)
        if 'html' in response.url:
            print("Error when downloading " + link)
            return False
        else:
            directoryPath = os.path.dirname(destination_path)
            try:
                if not os.path.exists(directoryPath):
                    os.makedirs(directoryPath)
            except Exception as ex:
                print('Error found in', link, 'for directory',directoryPath)
                print(ex)
            file = open(destination_path, 'wb')
            document = response.read()
            file.write(document)
            file.close()
            return True
    
    except Exception as ex:
        print('Error found in', link)
        print(ex)

def getListFiles(content_list, suffixes_list = None, filtered_dict=None):
    return [oFile for oFile in content_list if oFile.suffix in suffixes_list and (filtered_dict is None or filtered_dict.get(oFile.stem) is None)]

def exploreDirectory(data_dir, suffixes_list = [".pdf",".png",".jpg","jpeg"], filtered_dict=None):
    content_list = [Path(file_name) for file_name in os.listdir(data_dir)]
    #check if the main directory contains directories
    directory_list = [oFile for oFile in os.listdir(data_dir) if not os.path.isfile(oFile)]
    if len(directory_list) ==0:
        file_list = getListFiles(content_list)
    else:
        file_list = []
        for root, subdirs, files_ in os.walk(data_dir):
            content_list = [Path(os.path.join(root,file_name)) for file_name in files_]
            files = getListFiles(content_list, suffixes_list, filtered_dict)
            if len(files)>0:
                file_list.extend(files)
    return file_list
    
#@ray.remote
#def remote_apply_ocr_and_html(info):
#    return apply_ocr_and_html(info)

def apply_ocr_and_html(info): 
    error_list = {}
    file_list = info['data']
    lang_modules = info['modules']
    sample_response_list = {} 
    file_name = None
    base_list = {'words':[],  'pages':[], 'raw_text':{}, 'raw_corrected_text':{}, 'corrections':[], 'orphans':[]} #'bbox':[],
    for i_file_doc, fileDoc in enumerate(file_list):
        try:
            pdf_file_path = fileDoc['datafile']  
            split_type = fileDoc['split_type']
            detect_lang = fileDoc['detect_lang']
            page_start = fileDoc['page_start'] 
            page_end = fileDoc['page_end']
            #lang = fileDoc['lang'] 
            total_pages = 0
            file_name_dict = pdf_file_path + "__" +  str(page_start)  
            file_name =  Path(pdf_file_path).stem 

            if error_list.get(file_name) is not None:
                continue

            sample_response_list[file_name_dict] =  copy.deepcopy(base_list)
            try:
                with pikepdf.open(pdf_file_path) as pdf:
                    total_pages =  len(pdf.pages)
                    sample_response_list[file_name_dict]['total_pages'] = total_pages
                if page_end<0 or page_end > total_pages: page_end = total_pages
                gc.collect()
                images = convert_from_path(pdf_file_path, dpi=300, first_page=page_start+1, last_page=page_end)
                sample_response_list[file_name_dict]['last_page'] = page_start + len(images) 
            except Exception as ex:
                print('Error while opening the file ', pdf_file_path , '\n', ex)
                print(ex)
                continue


            is_encrypted = False
            print('Starting ', pdf_file_path, 'from page', page_start, 'to page', page_end, 'with ', total_pages, 'pages')
            
            #pdf_file_path = os.path.join(Path(pdf_file_path).parent , '035V5.pdf')
            #using pdf2toimage to get images from pdf

            
            read_pdf_file_path = pdf_file_path 
            #using fitz to read html to check if document is scanned or encoded, exception check if document is encrypted
            with fitz.open(pdf_file_path) as doc_fitz_temp:
                try:
                    page_fitz_temp = doc_fitz_temp[0]
                    del page_fitz_temp
                except ValueError as ex:
                    decripted__file_path = decryptPDF(pdf_file_path)
                    print('Decrypted for working ', pdf_file_path)
                    print(ex)
                    is_encrypted = True
                    if decripted__file_path is None:
                        print('ERROR. Nothing to decrypt, on ', pdf_file_path, ' ', ex) 
                        raise ex 
                    with fitz.open(decripted__file_path) as doc_fitz_temp: 
                        try:
                            page_fitz_temp = doc_fitz_temp[0]
                            del page_fitz_temp
                            read_pdf_file_path = decripted__file_path
                        except Exception as ex2:
                            print('ERROR, decrypted but with errors,  on ', pdf_file_path, '  ', ex2) 
                            continue
            
            doc_fitz = fitz.open(read_pdf_file_path)
            
            #using pytesseract to extract text, bbox and orientation information
            
            for i_page, image in enumerate(images): 
                page_list = []
                word_list = []
                width, height = image.size
                
                if  height > width:
                    orientation = 'P'
                else:
                    orientation = 'L'
                file_name_page = Path(pdf_file_path).stem + '_' + str(i_page+page_start)
                
                res_rotation = getPageTextRotating(image, file_name_page , detect_lang=detect_lang,lang_modules=lang_modules)

                if res_rotation['rotation'] == 90 or res_rotation['rotation'] ==270:
                    orientation = 'L'
                
                
                if res_rotation['status'] != 'Empty':
                    oLanguage = res_rotation['lang']
                    rotation = res_rotation['rotation']
                    #TODO :add all the languages that were discovered
                    ocr_df = pytesseract.image_to_data(res_rotation['image'], output_type='data.frame', config = '-l ' + oLanguage[0])  #config = r'-l ' + lang + ' --oem 3 --psm 6'
                    ocr_df = ocr_df.dropna().reset_index(drop=True)
                    raw_text_ocr = pytesseract.image_to_string(res_rotation['image'], config = '-l ' + oLanguage[0])
                    if raw_text_ocr =='\x0c' or len(raw_text_ocr)==0: continue
                    #oImage = Path(pdf_file_path)
                
                    float_cols = ocr_df.select_dtypes('float').columns
                    ocr_df = ocr_df.dropna().reset_index(drop=True)
                    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
                    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True) #replace empty strings with NaN values
                    ocr_df = ocr_df.dropna().reset_index(drop=True)

                    data_columns = ocr_df[['left', 'top', 'width', 'height','block_num','line_num','text','level', 'word_num','conf']]
                    
                    data = []
                    
                    for idx, row in data_columns.iterrows():
                        x, y, w, h, block_num, line_num, text, level, word_num, conf= tuple(row) 
                        data.append([ block_num, line_num, text, x, y, x + w, y + h, w, h, level, word_num, conf])
                        #               0           1       2    3  4    5      6    7  8   9        10       11     
                    if len(data) ==0: continue
                    if split_type == 'line':
                        blocks = {item_block[0]:{item_line[1]:0 for item_line in data if item_line[0]==item_block[0]} for item_block in data}  
                        text_lines = []
                        for block in blocks:
                            for line in blocks[block]:
                                text_data = [item for item in data if item[0]==block and item[1]==line] 
                                text_data.sort(key=lambda x:x[4], reverse=False)
                                word_in_text_indexes = {i:0 for i,item in enumerate(text_data)}
                                while 0 in word_in_text_indexes.values():
                                    first_word = [index for index in word_in_text_indexes if word_in_text_indexes[index] ==0]
                                    if len(first_word) >0:
                                        first_word = first_word[0]
                                        line_data = [(index,text_data[index]) for index in word_in_text_indexes if text_data[index][4]<text_data[first_word][6] and word_in_text_indexes[index]==0]
                                        text_data_line = [item[1] for item in line_data]
                                        index_data_line = [item[0] for item in line_data]
                                        text_data_line.sort(key=lambda x:x[3], reverse=False)
                                        
                                        try:
                                            text_ = ' '.join([item[2] for item in text_data_line])
                                        except TypeError:
                                            print('Error ', text_data_line, ' ----- ')
                                            print('Line data ', line_data)

                                        word_list.extend(text_.split(' '))
                                        text_x = min([item[3] for item in text_data_line])
                                        text_y = min([item[4] for item in text_data_line])
                                        text_xw = max([item[5] for item in text_data_line])
                                        text_yh = max([item[6] for item in text_data_line])  
                                        text_w =  text_xw - text_x
                                        text_h =   text_yh - text_y
                                        text_avg_h = round(np.average([item[8] for item in text_data_line])) 
                                        string_dic =  {'file_name':file_name,'page_number':i_page+page_start+1,'line_number':line,'text':text_, 'x':text_x,'y': text_y, 'x2':text_xw, 'y2':text_yh, 'w':text_w, 'h':text_h, 'avg_h': text_avg_h} 
                                        
                                        #text_lines.append(string_line)
                                        text_lines.append(string_dic)
                                        
                                        if len(line_data) == len(text_data):
                                            break

                                        index_data_line = [item[0] for item in line_data]
                                        for index_val in index_data_line:
                                            word_in_text_indexes[index_val] = 1
                                    else:
                                        break
                                
                    elif split_type == 'word': 
                        for data_word in data: 
                            word_list.extend(data_word[2])
                            text_line =  [data_word[2], data_word[3], data_word[4], data_word[5], data_word[6], data_word[5]-data_word[3], data_word[6]-data_word[4], data_word[8], width, height]
                            text_lines.append(text_line)
                    else:
                        raise Exception('split_type =' + split_type + ' not implemented.')

                    line_numbers = {item:i for i,item in enumerate(np.sort([item['y'] for item in text_lines]))}
                    for i,item in enumerate(text_lines):
                        text_lines[i]['line_number'] = line_numbers[item['y']]
                    text_lines.sort(key=lambda x:x['line_number'], reverse=False)
                
                    # normalize the bounding boxes
                    '''
                    text_boxes = []
                    for i,line in enumerate (text_lines):
                        normalized_bbox = normalize_box([line['x'], line['y'], line['x2'], line['y2']], width, height)
                        bbox_dic =  {'file_name':file_name,'page_number':i_page+page_start+1,'line_number':line,'n_y':normalized_bbox[0], 'n_y':normalized_bbox[1],'n_x2': normalized_bbox[2], 'n_y2':normalized_bbox[3], 'page_width':width, 'page_height':height} 
                                        
                        #text_lines[i] = '\t'.join([oImage.stem, str(i_page+page_start+ 1), str(i)]+[str(item) for item in text_lines[i]])
                        #bbox_line = '\t'.join([str(oImage.stem), str(i_page+page_start+1), str(i)]) + '\t' + str(normalize_box([line[1], line[2], line[3], line[4]], width, height))[1:-1]  + '\t'+ str(width) + '\t' +  str(height)
                        #text_boxes.append(bbox_line)
                        text_boxes.append(bbox_dic)

                    '''
                    
                    #correct OCR reading
                    replacement_list = []
                    page_fitz = doc_fitz[i_page+page_start]
                    if len(raw_text_ocr) > 5:
                        page_html = page_fitz.get_text("html")
                        html_content = BeautifulSoup(page_html, "html.parser")
                        try:
                            if len(text_lines) >0:
                                replacement_list, orphans_lines_list = correctOCRReading(html_content, text_lines)
                        except Exception as ex:
                            print('Error in parallel node:', ex)
                            print('Error on file: ', fileDoc)
                            raise ex
                    else:
                        print('possible blank page')

                    if len(replacement_list) > 0:
                        for replacement_item in replacement_list:
                            #oCompare = cleanSentences([{'text':text_lines[replacement_item['line_number']]['text']}])[0]['text']
                            if len(text_lines) > replacement_item['ocr_line_number']:
                                text_lines[replacement_item['ocr_line_number']]['text'] = replacement_item['html']
                            else:
                                print('Posible error: Replacement item\n', replacement_item)
                                print('Posible error: Text_lines\n', text_lines)

                    #check if the page is empty, scanned or encoded
                    content_type, read_confidence, number_images  = getHTMLInformation(html_content, page_html, word_list)

                    if is_encrypted:
                        encrypted = "D"
                    else:
                        encrypted = "N"

                    #list of information at page level
                    #page_list = '\t'.join([oImage.stem, str(i_page+page_start+1), str(width), str(height), orientation, str(rotation), oLanguage[0], str(oLanguage[1]), content_type, str(read_confidence), encrypted, str(number_images)])
                    page_dic = {'file_name':file_name, 'page_number': i_page+page_start+1, 'page_width': width,'page_height': height, 'orientation':orientation, 'angle': rotation, 
                        'language':oLanguage[0], 'lang_confidence': str(oLanguage[1]), 'content_type': content_type, 'read_confidence':read_confidence, 'encrypted': encrypted, 'number_images':number_images}
                    
                    #assert len(text_lines)==len(text_boxes)
                    if len(text_lines) >0: 
                        sample_response_list[file_name_dict]['words'].extend(text_lines) 
                        #sample_response_list[file_name_dict]['bbox'].extend(text_boxes) 
                        sample_response_list[file_name_dict]['raw_text'][Path(file_name_dict).stem + "__" + str(i_page+page_start+1)]=raw_text_ocr

                    sample_response_list[file_name_dict]['pages'].append(page_dic) 
                    
                    if len(replacement_list) > 0:
                        sample_response_list[file_name_dict]['corrections'].extend(replacement_list)
                    if len(orphans_lines_list) > 0:
                        sample_response_list[file_name_dict]['orphans'].extend(orphans_lines_list)

            doc_fitz.close()
            
            #add correct raw list: 
            page_consolidated_text = []
            for oPage in sample_response_list[file_name_dict]['pages']:
                page_number = oPage["page_number"]
                page_corrections = [item for item in sample_response_list[file_name_dict]['corrections'] if item['page_number']==page_number]
                base_text = sample_response_list[file_name_dict]['raw_text'][Path(file_name_dict).stem + "__" + str(page_number)]
                base_text = cleanSentences([{'text':base_text}])[0]['text'] 
                if len(page_corrections)>0:
                    for correction in page_corrections:
                        base_text = base_text.replace(correction['ocr'],correction['html'],1)
                page_consolidated_text.append(base_text)

            sample_response_list[file_name_dict]['raw_corrected_text'] = {'pages':page_consolidated_text, "number_pages":len(page_consolidated_text)}
            print('Finishing ', pdf_file_path)

        except Exception as ex:
            print('ERROR! Not finished ', pdf_file_path)
            print(pdf_file_path, ex)
            if sample_response_list.get(file_name_dict) is not None:
                sample_response_list.pop(file_name_dict)
            
            for file_path in sample_response_list:
                if Path(file_path).stem == file_name:
                    sample_response_list[file_path] = copy.deepcopy(base_list)
            
            error_list[file_name] = str(ex)

    return sample_response_list, error_list

#remote_processing = apply_ocr_and_html.remote() #ray.remote(apply_ocr_and_html)
#@ray.remote
def remote_process_group_by_document(info):
    consolidated_results = []
    for document_name in info['document_list']: 
        last_cia_document = info['last_cia_documents'].get(document_name)
        row_list_document = [item for item in info['row_list_document'] if item['file_name']==document_name]
        results = process_group_by_document(info['group_level'], row_list_document, document_name, info['filter_min_words'], info['additional_headers'], info['y_label_type'], last_cia_document)
        if len(consolidated_results)==0:
            consolidated_results = results
        else:
            for i_result, result in enumerate(results):
                consolidated_results[i_result].extend(result)
    return consolidated_results

def process_group_by_document(group_level, row_list_document, document_name, filter_min_words, additional_headers, y_label_type, last_cia_document):
    #last_page = row_list_page[0]['page_number']
    temp_text_lang = []
    temp_text = []
    temp_all_metadata = []
    temp_all_words = []
    temp_all_labels = []
    last_enumerator = ''
    
    if group_level in ['paragraph', 'first_paragraph']:
        accumulated_text = '' 
        if last_cia_document == document_name:
        #if self.dataset_last_documents.get(row_list_page[-1]['company_id']) == document_name:
            last_year_doc = True
        else:
            last_year_doc = False

        for row_number, row in enumerate(row_list_document):
            enumerators = getEnumeratorsLine(row['text'])
            if len(enumerators) >0: #current line is enumerator
                if len(accumulated_text) >0 and len(last_enumerator)>0:
                    if filter_min_words is None or filter_min_words ==0 or len([item for item in accumulated_text.split(' ') if len(item)>2]) > filter_min_words:                                                       
                        metadata_info = None
                        if row.get('document_year') is not None and row.get('company_id') is not None: 
                            metadata_info = {'document': document_name, 'page':row['page_number'],'year': row['document_year'], 'company_id':row['company_id'], 'last_year_doc':last_year_doc}    
                        else:
                            metadata_info = {'document': document_name, 'page':row['page_number']}

                        if len(additional_headers)>0:
                            metadata_info.update({item:row[item] for item in additional_headers})
                        
                        temp_all_metadata.append(metadata_info) # self.train.all_metadata.append(metadata_info)
                        temp_all_words.append(accumulated_text.lower().strip()) #self.train.all_words.append(accumulated_text.lower().strip()) 
                        temp_text_lang.append(row['language'])
                        temp_text.append(accumulated_text)
                        if y_label_type == 'group':
                            temp_all_labels.append(last_enumerator.lower())
                        else:
                            temp_all_labels.append(row_list_document[row_number-1][y_label_type].lower())
                        last_enumerator = ''

                temp_string =  row['text'][max([len(item) for item in enumerators]):].strip()
        
                if len(temp_string) > 0 and not hasNumbers(temp_string):
                    last_enumerator = temp_string
                else:
                    last_enumerator = ""
                accumulated_text = ""
            else: #Previous was enumerator, but the current one is not.
                if row_number>1 and row_list_document[row_number-1]['text'][-1] == "." and row_list_document[row_number]['text'][0].upper() == row_list_document[row_number]['text'][0]:
                    accumulated_text += "\n\n" + row['text'] 
                else:
                    accumulated_text += " " + row['text'] 
    else:
        if group_level in  ['line', 'document']:
            page_list = {0:0}
        elif group_level == 'page':
            page_list = {item['page_number']:0 for item in row_list_document} #rows_list if item['file_name']==document_name} 

        for page_number in page_list:
            #row_list_document = [row for row in rows_list if row['file_name']==document_name and (page_number==0 or row['page_number']==page_number)]
            row_list_page = [row for row in row_list_document if page_number==0 or row['page_number']==page_number]
            
            #if self.dataset_last_documents.get(row_list_page[-1]['company_id']) == document_name:
            if last_cia_document == document_name:
                last_year_doc = True
            else:
                last_year_doc = False

            if group_level in  ['document', 'page']:
                accumulated_text_list = ['\n'.join([item['text'] for item in row_list_page])]
                label_list = [row_list_page[-1].get(y_label_type)]
                row_info_list = [{'document_year':row_list_page[-1].get('document_year'), 'company_id':row_list_page[-1].get('company_id')} ]
            elif group_level == 'line':
                accumulated_text_list =  [item['text'] for item in row_list_page]
                label_list = [item.get(y_label_type) for item in row_list_page]
                row_info_list = [(item.get('document_year'),item.get('company_id')) for item in row_list_page]

            current_page = row_list_page[-1]['page_number']
            current_language = Counter([item.get('language') for item in row_list_page]).most_common(1)[0][0]

            for i, accumulated_text in enumerate(accumulated_text_list): 
                label_row = label_list[i].lower() if label_list[i] is not None else label_list[i]

                temp_all_words.append(accumulated_text.lower())
                temp_all_labels.append(label_row)

                if row_info_list[i].get('document_year') is not None and row_info_list[i].get('company_id') is not None:     
                    temp_all_metadata.append({'document': document_name, 'page':current_page,'year': row_info_list[i]['document_year'], 'company_id':row_info_list[i]['company_id'], 'last_year_doc':last_year_doc})
                else:
                    temp_all_metadata.append({'document': document_name, 'page':current_page})

                temp_text_lang.append(current_language)
                temp_text.append(accumulated_text) 

                #if number_records is not None and len(temp_text) >= number_records:
                #    break
    return temp_all_metadata, temp_all_words, temp_text_lang, temp_text, temp_all_labels

def remote_tokenize_text(info):
    consolidated_results = {}
    for i_sample, sample in enumerate(info['all_data']['all_words']): 
        label_sample = info['all_data']['all_labels'][i_sample]
        results =  tokenize_text(info['tokenizer'], sample, label_sample, info['sequence_strategy'], info['max_sequence_length'], info['trim_begining'], info['labels2idx'])
        if len(consolidated_results)==0:
            consolidated_results = results
            consolidated_results['all_metadata'] = [info['all_data']['all_metadata'][i_sample]]
            consolidated_results['all_labels'] = [label_sample]
            consolidated_results['text_lang'] = [info['all_data']['text_lang'][i_sample]]
        else:
            for result_type in results:
                consolidated_results[result_type].extend(results[result_type])
            consolidated_results['all_metadata'].append(info['all_data']['all_metadata'][i_sample]) 
            consolidated_results['all_labels'].append(label_sample)
            consolidated_results['text_lang'].append(info['all_data']['text_lang'][i_sample])

    return consolidated_results


def tokenize_text(tokenizer, sample, label, sequence_strategy, max_sequence_length, trim_begining, labels2idx):
    featured_data = {'labels_ids':[],'text':[],'words':[],'input_ids':[],'token_type_ids':[],'attention_mask':[], 'encoded':[]}
    sample_tokenized =  tokenizer.tokenize(sample) 
    enconded_sample = tokenizer(sample).data
    if label is None:
        featured_data['labels_ids'].append(None)
    else:
        featured_data['labels_ids'].append(labels2idx[label])
    if len(sample_tokenized) - 2 > max_sequence_length:                
        if sequence_strategy == 'truncate_max_seq': #Truncate dataset
            if trim_begining:
                start = max_sequence_length-1
                end = -1
            else:
                start = 1
                end = max_sequence_length-1
            featured_data['text'].append(sample)
            featured_data['words'].append([tokenizer.cls_token] + sample_tokenized[start:end] +  [tokenizer.sep_token])
            featured_data['input_ids'].append(tokenizer.cls_token_id + enconded_sample['input_ids'][start:end] + tokenizer.sep_token_id)
            featured_data['token_type_ids'].append(enconded_sample['token_type_ids'][0]+ enconded_sample['token_type_ids'][start:end] + enconded_sample['token_type_ids'][0] )
            featured_data['attention_mask'].append(enconded_sample['attention_mask'][0] + enconded_sample['attention_mask'][start:end] + enconded_sample['attention_mask'][0] )
        elif sequence_strategy == 'document_batch':
            featured_data['text'].append(sample)
            featured_data['words'].append(sample_tokenized)
            featured_data['encoded'].append(enconded_sample) 
        else:
            raise Exception('Sequence strategy not yet implemented.')

    else:
        featured_data['text'].append(sample)
        featured_data['words'].append(sample_tokenized)  
        featured_data['encoded'].append(enconded_sample)
    
    return featured_data

def getRayPool(total_workers):
    oPool = Pool()
    print("Nodes in the Ray cluster:", ray.nodes())
    if os.environ.get('ip_head') is not None:
        ray.init(address=os.environ["ip_head"], num_cpus = total_workers*ray.nodes(), ignore_reinit_error=True)
    else:
        ray.init(num_cpus = total_workers, ignore_reinit_error=True)

    return oPool

def distribute(distrib_tool, dataset_batch, function_to_distribute, total_workers=1):
    try:
        if distrib_tool =='ray': 
            oPool = getRayPool(total_workers)
            futures = [l_batch for l_batch in dataset_batch]
            results = oPool.map(function_to_distribute, futures) #apply_ocr_and_html 
        else:
            oPool = mp.Pool(total_workers, maxtasksperchild=1)
            results = oPool.map(function_to_distribute, dataset_batch)#apply_ocr_and_html 
            oPool.close()
    except Exception as ex:
        print('EXCEPTION. Distribution with', distrib_tool, '. Number of workers: ', len(dataset_batch))
        print(ex) 
        raise ex
    return results

def normalize_box(box, width, height):
     return [
         int(1000 * (box[0] / width)),
         int(1000 * (box[1] / height)),
         int(1000 * (box[2] / width)),
         int(1000 * (box[3] / height)),
     ]


def getPageTextRotating(image, image_name, detect_lang=True, lang=None, lang_modules=None): 
    module_stopwords = lang_modules['STOP_WORDS']
    module_words =  lang_modules['WORDS']
    unified_stop_words = {}
    if lang is not None:
        unified_stop_words = {item:0 for item in module_stopwords[lang]}
    else:
        for lang_ in module_stopwords:
            unified_stop_words.update({item:0 for item in module_stopwords[lang_]})
    
    directory_temp = 'temp'
    image_name = os.path.join(directory_temp,'temp_' + image_name + '_' + str(random.randint(0,1000))  + '.jpg')
    if not os.path.exists(directory_temp):
        os.makedirs(directory_temp)
    image.save(image_name)
    empty_page = False

    try:
        osd_resp = pytesseract.image_to_osd(image_name, config ='--psm 0 -c min_characters_to_try=5 ' + '-l ' + '+'.join(list(langs.keys())), output_type=Output.DICT)
        orientation = osd_resp['orientation']
    except Exception as ex:
        empty_page = True
        orientation = 0
        print('Error while using pytesseract, possible blank page' + image_name)
        print(ex)
        
    if os.path.exists(image_name):
        os.remove(image_name)
    
    status = 'OK'
    if orientation !=0:
        image_temp = image.rotate(orientation, expand=True)
        status = 'Rotated'
    else:
        image_temp = image
    
    text_ = pytesseract.image_to_string(image_temp, config = '-l ' + '+'.join(list(langs.keys())) + ' --oem 3 --psm 6')  #config = r'-l ' + lang + ' --oem 3 --psm 6'
    text_ = text_.lower()
    words = [w for line in text_.split('\n') for w in line.split(' ') if len(w.strip())>0]

    if len(words) == 0:
        return {'image': image_temp, 'rotation': 0, 'status':'Empty', 'lang': ['',0]}
    
    if len(words) >0:
        if detect_lang:
            lang = getLanguageByWords(words, module_stopwords)
            if lang[1]<0.4:
                lang = getLanguageByWords(words, module_words)
                if lang[1]==0:
                    lang = getLanguageByWords(words, module_words)
        else:
            lang = [lang, 2]



   
    return {'image': image_temp, 'rotation': orientation, 'status':status, 'lang': lang}
    
        

def download_lang_dictionaries():    
    for lang in PRETRAINED_VOCAB_FILES_MAP.keys():
        file_url = PRETRAINED_VOCAB_FILES_MAP[lang]
        file_name = file_url.split('/')[-1] 
        directory_location = 'dicts/'
        if not os.path.exists(directory_location):
            os.makedirs(directory_location)

        file_location = os.path.join(directory_location, lang + '_' + file_name)
        if not os.path.exists(file_location):
            response = requests.get(file_url)
            if Path(file_url).suffix == '.json':
                with open(file_location, 'w', encoding='utf-8') as oFile:
                    json.dump(json.loads(response.content.decode('utf-8')), oFile)
            else:
                with open(file_location, "w", encoding='utf-8') as oFile:
                    oFile.write(response.content.decode('utf-8'))

def get_wordsmodules():
    directory_location = 'dicts/'
    langModules_words = {}
    for lang in PRETRAINED_VOCAB_FILES_MAP.keys():
        file_url = PRETRAINED_VOCAB_FILES_MAP[lang]
        file_name = file_url.split('/')[-1] 
        file_location = os.path.join(directory_location, lang + '_' + file_name)
        
        if Path(file_location).suffix == '.json':
            with open(file_location, "r") as oFile:
                dict_words = json.load(oFile)
                dict_words = {item.split('<')[0] for item in dict_words}  
                langModules_words[lang] = {item for item in dict_words if len(item.strip())>0}  
        else:
            with open(file_location, "r", encoding='utf-8') as oFile:
                dict_words = oFile.readlines()
                dict_words = {word.split('\n')[0]:0 for word in dict_words}
                langModules_words[lang] = {item for item in dict_words if len(item.strip())>0}
 
    return langModules_words

def get_stopwordsmodules(lang=None):
    if lang is not None: 
        lang_values = list(langs.keys())[0]
    lang_values = list(langs.keys())
    langModules = {language: importlib.import_module("spacy.lang." + langs[language] + ".stop_words") for language in lang_values}
    langModules_words = {language:langModules[language].STOP_WORDS for language in langModules}
    return langModules_words

def getLanguageByWords(tokenized_text, words_modules): 
    langs_eq = list(langs.keys())
    langDicts = {}
    ratios_stopwords = []

    total_words = len(tokenized_text)
    confidence = 0 
    for lang in list(langs.keys()): 
        langDicts[lang] = set(words_modules[lang])
        for lang2 in list(langs.keys()):
            if lang2 != lang:
                langDicts[lang] -= set(words_modules[lang2])

    if total_words >0:
        for lang in list(langs.keys()):
            ratios_stopwords.append(len([word for word in tokenized_text if word in langDicts[lang]])/total_words)

        if np.sum(ratios_stopwords) >0:
            confidence = int(np.max(ratios_stopwords) / np.sum(ratios_stopwords)*100)/100
        if confidence ==0: #check how to improve this, create a dictionary of words that falls in this category and then organize per language.
            return [langs_eq[0], confidence]
        return [langs_eq[np.argmax(ratios_stopwords)], confidence]
    else: #Blank page, returning default language and confidence = 0
        return [langs_eq[0],  0]

def cleanSentences(list_sentences): 
    for i, sentence in enumerate(list_sentences):
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("‘","'")
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("—","-")
        list_sentences[i]['text'] = list_sentences[i]['text'].replace("’","'")
        if sentence.get('text_left') is not None and list_sentences[i]['text_left'] != list_sentences[i]['text'] :
            list_sentences[i]['text_left'] = list_sentences[i]['text'] 
    return list_sentences

def correctOCRReading(html_content, ocr_text_lines): 
    max_lev_ratio = 0.1

    list_words_html = [word.replace(u'\xa0', u' ') for line in html_content.text.split('\n') if len(line.split())>0 for word in line.split(' ') if len(word.strip())>0]
    list_words_html = [item for line in list_words_html for item in line.split(' ')]
   

    html_lines = [{'x':item['style'].split(";"),'text':item.text} for item in  html_content.find_all('p') if len(item)>0]
    html_lines = cleanSentences(html_lines)
    
    orphans_lines_list = []
    orphans_lines_list_last = []
    replacement_list = []
    ocr_lines = []
    
    if len(html_lines)>0:
        for i_line, html_line in enumerate(html_lines):
            features_style = {item.split(':')[0]:item.split(':')[1] for item in html_line['x']}
            html_lines[i_line]['y'] = int(features_style['top'].replace('pt','').split(".")[0])
            html_lines[i_line]['x'] = int(features_style['left'].replace('pt','').split(".")[0])
            html_lines[i_line]['matched'] = False

        #min_ratio_diff =0.03
        html_lines.sort(key = lambda x:(x['y'], x['x'])) 
        
        for html_line_single in html_lines: 
            if  html_line_single.get('text_line') is None:
                neighbors_list = [(item['x'],i, item) for i, item in enumerate(html_lines) if len(item['text'])>0 and abs(item['y']-html_line_single['y']) <8 ]
                neighbors_list.sort()
                text = ' '.join([item[2]['text'] for item in neighbors_list]) 
                indexes_line = [item[1] for item in neighbors_list]
                for o_neightborg in neighbors_list: 
                    html_lines[o_neightborg[1]]['text_line'] = text
                    html_lines[o_neightborg[1]]['text_line_indexes'] = indexes_line

        
        ocr_lines = [{'pos': i,'x':item['x'],'text':item['text'],'matched':False,'text_left':item['text'],'y':item['y'],'y_html':[]} for i,item in enumerate(ocr_text_lines)]
        ocr_lines = cleanSentences(ocr_lines)
        ocr_line_numbers_temp = [[item['y'],i] for i,item in enumerate(ocr_text_lines)]
        ocr_line_numbers_temp.sort()
        
        for i_line, html_line in enumerate(html_lines):
            is_orphan = True
        
            space = ''
        
            if html_line['matched']:
                continue
        
            for o_line in range(0, len(ocr_lines)): 
                if not ocr_lines[o_line]['matched']:
                    ocr_text_left = space + ocr_lines[o_line]['text_left'].strip() + space
                    html_text = space + html_line['text'].strip() + space
                    html_text_full =  space + html_line['text_line'].strip()  + space
                    if ocr_text_left.lower() == html_text_full.lower() or  ocr_text_left.lower() == html_text.lower(): #full direct match
                        is_orphan=False
                        ocr_lines[o_line]['matched'] = True
                        ocr_lines[o_line]['text_left'] = ''   
                        ocr_lines[o_line]['y_html'].append(html_line['y'])
                        if ocr_text_left.lower() == html_text_full.lower() :
                            for index_h in html_line['text_line_indexes']:
                                html_lines[index_h]['matched'] = True 
                        else:
                            html_lines[i_line]['matched'] = True
                        break
                    else:
                        ocr_text_full =  space + ocr_lines[o_line]['text'].strip() + space
                        lev_dist = lev(html_text_full.lower(), ocr_text_full.lower())
                        max_len = max(len(html_text_full), len(ocr_text_full))
                        if lev_dist / max_len < max_lev_ratio:
                            is_orphan=False
                            addToReplacementList(replacement_list, ocr_text_lines[0]['file_name'],page_number=ocr_text_lines[o_line]['page_number'],ocr_text=ocr_text_full.strip(),html_text=html_text_full.strip(), ocr_position=o_line, html_position=i_line)
                            ocr_lines[o_line]['text_left'] = ''
                            ocr_lines[o_line]['matched'] = True
                            ocr_lines[o_line]['y_html'].append(html_line['y'])
                            for index_h in html_line['text_line_indexes']:
                                html_lines[index_h]['matched'] = True
                            break
                        else:
                            new_text = (ocr_text_left.lower()).replace(html_text.lower(),'')
                            diff_new_text = len(ocr_text_left) - len(html_text)
                            if len(new_text)<=diff_new_text:    
                                if lev_dist / max_len < max_lev_ratio*2: #too much noise but still belongs to the line
                                    if diff_new_text > 0:                      #partial match    
                                        if len(new_text)==diff_new_text:                      
                                            is_orphan=False
                                            ocr_lines[o_line]['text_left'] = ocr_lines[o_line]['text_left'].lower().strip().replace(html_line['text'].lower().strip(),'').strip()
                                            ocr_lines[o_line]['y_html'].append(html_line['y'])
                                            html_line['matched'] = True
                                            break
                                        else:
                                            if len(new_text)<=diff_new_text: #replace the first coincidence                 
                                                is_orphan=False
                                                if html_text in ocr_text_left:
                                                    index_coincidence = (ocr_text_left).index(html_text)
                                                    ocr_lines[o_line]['text_left'] = ocr_lines[o_line]['text_left'].strip()[:index_coincidence] + ' ' + ocr_lines[o_line]['text_left'].strip()[index_coincidence+len(html_text):]
                                                    ocr_lines[o_line]['y_html'].append(html_line['y'])
                                                    html_line['matched'] = True
                                                    break
                                            else: #there is no coincidence because the word is part of a bigger one, with no spaces
                                                is_orphan=False
                                                ocr_text_left = ocr_lines[o_line]['text_left'].lower().strip()
                                                html_text = html_line['text'].lower().strip()
                                                if html_text in ocr_text_left:
                                                    index_coincidence = (ocr_text_left).index(html_text)
                                                    ocr_lines[o_line]['text_left'] = ocr_lines[o_line]['text_left'].strip()[:index_coincidence] + ' ' + ocr_lines[o_line]['text_left'].strip()[index_coincidence+len(ocr_text_left):]
                                                    ocr_lines[o_line]['y_html'].append(html_line['y'])
                                                    html_line['matched'] = True
                                                    print('POSSIBLE ERROR: \n\nhtml:\n',html_text,'\n\nocr left:\n',ocr_text_left)
                                                    break
            if is_orphan:
                orphans_lines_list.append(html_line)

        #correct orphans
        orphans_lines_list = [item for item in orphans_lines_list if len(item['text'].strip())>0]
        filtered_ocr_lines = [item for item in ocr_lines if not item['matched']] 
        
        
        orphans_lines_list_line = []
        for orphan_html in orphans_lines_list:
            if len(orphan_html['text'].strip()) > 0:
                neighbors_list = [(item['x'],i, item,item['y']) for i, item in enumerate(orphans_lines_list) if len(item['text'])>0 and not item['matched'] and abs(item['y']-orphan_html['y']) <10 ]
                neighbors_list.sort()
                text = ' '.join([item[2]['text'] for item in neighbors_list])
                orphans_lines_list_line.append({'x': neighbors_list[0][0], 'text': text,'y':neighbors_list[0][2]['y']})
                for o_neightborg in neighbors_list:
                    orphans_lines_list[o_neightborg[1]]['matched'] = True
                    orphans_lines_list[o_neightborg[1]]['text'] = ''
        
       
        for i_item, ocr_item in enumerate(filtered_ocr_lines):  
            if len(ocr_item['text'])>0: 
                neighbors_list = [(item['x'],i, item) for i, item in enumerate(filtered_ocr_lines) if len(item['text'])>0 and abs(item['y']-ocr_item['y']) <30]
                if len(neighbors_list)>1:
                    neighbors_list.sort()
                    text = ' '.join([item[2]['text_left'] for item in neighbors_list]) 
                    indexes_line = [(item[1], item[2]['text_left']) for item in neighbors_list]
                    ocr_item['text_left'] = text
                    ocr_item['text_line_indexes'] = indexes_line
                    for o_neightborg in neighbors_list:
                        if  o_neightborg[1] != i_item:
                            filtered_ocr_lines[o_neightborg[1]]['text'] = ''
                else:
                    ocr_item['text_line_indexes'] = neighbors_list[0][1]
        
        for orphan_html in orphans_lines_list_line:
            found_pair = False
            for orphan_ocr in filtered_ocr_lines:
                if len(orphan_ocr['y_html']) ==0 and len(orphan_ocr['text'])>0 and not orphan_ocr['matched']:
                    prev_html_y, next_html_y = 0, 0
                    prev_html_list = [item['y_html'] for item in ocr_lines[:orphan_ocr['pos']] if len(item['y_html'])>0 and orphan_ocr['y']-5 >  item['y']]
                    if len(prev_html_list) >0:
                        prev_html_y = max(prev_html_list[-1])
                    next_html_list = [item['y_html'] for item in ocr_lines[orphan_ocr['pos']+1:] if len(item['y_html'])>0 and orphan_ocr['y']+5 <  item['y']]
                    if len(next_html_list) >0:
                        next_html_y = min(next_html_list[0])
                    if orphan_html['y'] > prev_html_y-1 and orphan_html['y'] < next_html_y+1:
                        if  orphan_html['text'].strip().lower() != orphan_ocr['text_left'].strip().lower():
                            addToReplacementList(replacement_list, ocr_text_lines[0]['file_name'],page_number=ocr_text_lines[o_line]['page_number'], ocr_text=orphan_ocr['text_left'].strip(), html_text=orphan_html['text'].strip(), ocr_position=orphan_ocr['pos'], html_position=i_line) 
                        found_pair = True
                        orphan_ocr['matched'] = True
                        break

                elif len(orphan_ocr['y_html'])>0 and abs(np.average(orphan_ocr['y_html']) - orphan_html['y']) <10:
                    if abs(len(orphan_ocr['text_left'])- len(orphan_html['text'])) < 5:
                        addToReplacementList(replacement_list, ocr_text_lines[0]['file_name'],page_number=ocr_text_lines[o_line]['page_number'],ocr_text=orphan_ocr['text_left'].strip(), html_text=orphan_html['text'].strip(),ocr_position=orphan_ocr['pos'], html_position=i_line)
                        found_pair = True
                        break

            if not found_pair and len(ocr_text_lines)>0:
                orphan_html['file_name'] = ocr_text_lines[0]['file_name']
                orphans_lines_list_last.append(orphan_html)

    return replacement_list, orphans_lines_list_last

def addToReplacementList(replacement_list, file_name, page_number, ocr_text, html_text, ocr_position, html_position):
    lev_dist = abs(lev(ocr_text,html_text))
    if lev_dist >0:
        replacement_list.append({'file_name':file_name,'page_number':page_number, 'ocr':ocr_text, 'html':html_text,'ocr_line_number':ocr_position, 'html_line_number': html_position,'distance':lev_dist})   
    return replacement_list                    

def getHTMLInformation(html_content, html_pageText, ocr_word_list):
    confidence,number_images = 0, 0
    
    number_words_ocr = len(ocr_word_list)
    
    list_words_html = [word.replace(u'\xa0', u' ') for line in html_content.text.split('\n') if len(line.split())>0 for word in line.split(' ') if len(word.strip())>0]
    list_words_html = [item for line in list_words_html for item in line.split(' ')]

    #count number of images in the document
    imageList = []
    if len(html_content.find_all("p"))>0: 
        for i_line, line in enumerate(html_content.find_all("img")):
            if 'src=' in str(line):
                number_images += 1  
                imageList.append(i_line) 
    #if no text was read before, is Empty
    if number_words_ocr == 0:
        confidence = 1
        if len(list_words_html) > 0:
            confidence = 0.5
        return "E", confidence, number_images #Empty, if 0.5 means that there are words, not completely empty

    ratio_words_html_ocr = len(list_words_html)/number_words_ocr
    ratio_words_ocr_html = 0

    if ratio_words_html_ocr >0:
        ratio_words_ocr_html = 1/ratio_words_html_ocr

    #check if there is encoded text
    if '&#xfffd;&#xfffd;&#xfffd' in html_pageText:
        no_encoded_text = html_pageText.replace('&#xfffd','')
        no_encoded_text = no_encoded_text.replace(';',' ')
        no_encoded_html_content = BeautifulSoup(no_encoded_text, "html.parser")
        number_words_html = sum([len(line.split(' ')) for line in no_encoded_html_content.text.split('\n') if len(line.strip())>0])
        if number_words_html > number_words_ocr:
            number_chars_html = sum([len(line.replace(' ','')) for line in no_encoded_html_content.text.split('\n') if len(line.strip())>0])
            number_chars_words = sum([len(word) for word in ocr_word_list])
            confidence = int((number_chars_words-number_chars_html)/number_chars_words*100)/100 
            if np.floor(confidence*100) == 0:
                confidence = int(len(html_pageText.split('&#xfffd'))/number_chars_words*10000)/10000
        else:
            confidence = int((number_words_ocr-number_words_html)/number_words_ocr*100)/100 #some quantities are less than zero
        return 'X', confidence, number_images #Encoded, confidence is how much of the document is encoded
  
    if ratio_words_html_ocr < 0.5:
        confidence = int((number_words_ocr - len(list_words_html))/number_words_ocr*100)/100
        return 'S', confidence, number_images  #Scanned, confidence of how much of the document is scanned
    else:
        confidence = int(min(ratio_words_html_ocr,ratio_words_ocr_html)*100)/100 
        if ratio_words_html_ocr < 0.6 or ratio_words_ocr_html < 0.6:
            return 'C', confidence, number_images #Bad reading, how much of the document could have a bad reading
    
        if ratio_words_html_ocr < 0.75 or ratio_words_ocr_html < 0.75:
            return 'B', confidence, number_images #Possible Bad Reading 
    
        return 'R', min(confidence,1), number_images #Readable PDF

    
def decryptPDF(fileName):
    oPathFile =  Path(fileName)
    destinationDirectory = os.path.join(oPathFile.parent, "decrypted")
    destination = os.path.join(destinationDirectory, fileName)

    if os.path.exists(destination):
        return destination

    if not os.path.exists(destinationDirectory):
        os.mkdir(destinationDirectory)
    
    try:
        with pikepdf.open(fileName) as pdf:
            pdf.save(destination)
            return destination
    except:
        return None


def getEnumeratorsLine(text):
    note_prefixes = ['note' ,'punkt', 'erläuterung','anmerkung']
    word = r'(?:'
    for wordFilter in note_prefixes:
        word += wordFilter + "|"
    word +=r'\s)*'

    enumeratorRegex1 = r'^[•>✓.\-–—_\-\*�]?' + word + r'[ivx]?[0-9a-j]?[.]?[0-9]{0,2}[ivx]*[.]?[0-9]?\s?[).\-–_:\-—\*]{0,2}\s'
    enumeratorRegex2 = r'^[•>✓.\-–—_\\*�]?' + word + r'[0-9]{1,2}[.]?[0-9]{0,2}[.]?[0-9]{0,2}[).\-–—_:\-\*\s]{1,2}'
    enumeratorRegex3 = r'^[•>✓.\-–—_\-\*�]\s?' + word
    enumeratorRegex4 = r'^([0-9a-f][.-][0-9a-f]?[.-]?)+'
    results3 = re.findall(enumeratorRegex3, text.lower().strip())
    results1 = re.findall(enumeratorRegex1, text.lower().strip())
    results2 = re.findall(enumeratorRegex2, text.lower().strip())
    results4 = re.findall(enumeratorRegex4, text.lower().strip())

    results1.extend(results2)
    results1.extend(results3)
    results1.extend(results4)
    if len(text) >0:
        first_char = text[0]
        if ord(first_char) in [65533, 61623]:
            results1.extend(first_char)

    return results1

def hasNumbers(text):
    numberRegex = "[0-9]+[.,']{0,1}[0-9]*"
    results = re.findall(numberRegex, text)
    return False if len(results) ==0 else True

def getProportionNumbers(text):
    numberRegex = "[0-9]+[.,']{0,1}[0-9]*"
    results = re.findall(numberRegex, text)
    number_words = len(text.split(' '))
    if number_words > 0:
        return len(results)/number_words
    return 0

class FDExTGenerator():
    
    modules = {}
       
    def __init__(self, data_dir, output_dir, split_type, filetype, filter_list_path=None, multifile_option='batch', check_loaded_docs =True, simple_load=False):
        self.output_dir = output_dir  
        self.split_type = split_type
        self.data_dir = data_dir
        self.filter_files = None
        self.file_type = filetype
        
        if multifile_option in ['document','page']:
            self.consolidate_page_info = False
        else:
            self.consolidate_page_info = True

        self.createFileNames(filetype, multifile_option)
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        self.multifile_option = multifile_option

        print('Documents are going to be generated by', multifile_option)
        
        if check_loaded_docs:
            print('Starting FDExTGenerator. Reading processed documents')
            self.loaded_docs = self.get_loaded_and_working_docs(print_info=False)
            if len(self.loaded_docs)>0: print('Already loaded ',  len(self.loaded_docs), 'documents in the dataset')
        else:
            print('Not checking loaded docs')
            self.loaded_docs = []

        if not simple_load:
            if filter_list_path is not None:
                if Path(filter_list_path).suffix=='.json':
                    with open(filter_list_path, 'r') as f:
                        filter_files =  json.load(f)
                else:
                    with open(filter_list_path, 'r') as f:
                        filter_files =  f.readlines()
                    filter_files = [item.split('\n')[0] for item in filter_files]
                self.filter_files = {item:0 for item in filter_files}

            download_lang_dictionaries()
            self.modules['WORDS'] = get_wordsmodules()
            self.modules['STOP_WORDS'] = get_stopwordsmodules()
    

    def get_loaded_and_working_docs(self, print_info=False):
        loaded_docs_dict = {}
        working_docs_dict = {}
        if print_info: print('multifile_option', self.multifile_option, 'self.pageInfo_filename',self.pageInfo_filename)
        if self.multifile_option is not None:
            oFileInfo = Path(self.pageInfo_filename)
            oDirectory = oFileInfo.parent
            file_list = [Path(os.path.join(str(oDirectory), file_name)) for file_name in os.listdir(oFileInfo.parent) if self.pageInfo_suffix in file_name and oFileInfo.name !=file_name] 
            #if print_info: print('File_list', file_list)
            for file_name in file_list: 
                loaded_docs = self.read_list_loaded_docs(self.file_type, file_name)
                if print_info: print('Directory',oDirectory,'file_name', file_name, 'Total docs', len(loaded_docs))
                if len(loaded_docs)>0: loaded_docs_dict.update(loaded_docs) 
            
            oDocTemp_suffix = self.get_temp_doclist_name().split(".")[-1]
            file_list = [Path(os.path.join(str(oDirectory), file_name)) for file_name in os.listdir(oFileInfo.parent) if oDocTemp_suffix in  file_name] 
            for file_name in file_list: 
                oDate = datetime.strptime(file_name.stem, "%Y_%m_%d_%H_%M_%S_%f")
                oDiff = (datetime.now()-oDate)
                if oDiff.days >=2: #removing old temporal files (>=2 days) 
                    os.remove(file_name)
                else:
                    loaded_docs = self.read_list_loaded_docs('json', file_name)
                if len(loaded_docs)>0: working_docs_dict.update(loaded_docs) 
                
            if len(working_docs_dict)>0 and print_info: print('Number of documents in progress',  len(working_docs_dict))
        else:    
            if os.path.exists(self.pageInfo_filename):   
                loaded_docs_dict = self.read_list_loaded_docs(self.file_type,self.pageInfo_filename)
               
        if len(loaded_docs_dict)>0 and print_info: print('Already loaded ',  len(loaded_docs_dict), 'documents in the dataset') 
        if len(working_docs_dict)>0: loaded_docs_dict.update(working_docs_dict)

        return loaded_docs_dict

    def read_list_loaded_docs(self, file_type, file_name):
        loaded_docs = []
        if file_type =='json':
            with open(file_name, ) as f:
                loaded_pages = json.load(f)
                loaded_docs= {doc['file_name']:0 for doc in loaded_pages} 
            
        elif file_type == 'txt':
            with open(file_name, 'r') as f:
                loaded_pages =  f.readlines()
                if len(loaded_pages) > 0: 
                    loaded_pages = [page.split('\t')[:2] for page in loaded_pages] 
                    loaded_docs = {doc[0]:0 for doc in loaded_pages}
        
        return loaded_docs
    
    def get_last_batch_number(self, directory_path):
        last_batch_number = -1
        oFileInfo =  Path(self.pageInfo_filename)
        file_list = [Path(os.path.join(directory_path, file_name)) for file_name in os.listdir(directory_path) if oFileInfo.stem in file_name and oFileInfo.name !=file_name] 
        
        for file_name in file_list:
            suffix_number = int(file_name.stem.split("_")[-1][3:]) 
            if last_batch_number < suffix_number: last_batch_number = suffix_number

        return last_batch_number

    def createFileNames(self, filetype="json", multifile_option=None):
        if multifile_option is None or len(multifile_option)==0 or multifile_option == "batch":
            multifile_option = ""
        else:
            multifile_option = "_#" + multifile_option

        self.pageInfo_suffix =  '_page_info'
        self.text_filename = os.path.join(self.output_dir, 'text_' + self.split_type + multifile_option +'.' + filetype)  
        self.raw_text_filename = os.path.join(self.output_dir, 'raw_text_' + self.split_type +  multifile_option + '.json')  
        self.raw_text_corrected_filename = os.path.join(self.output_dir, 'raw_correct_text_' + self.split_type +  multifile_option + '.json')  
        #self.bbox_filename = os.path.join(self.output_dir, 'text_' + self.split_type +  multifile_option + '_bbox.' + filetype) 
        self.format_filename = os.path.join(self.output_dir, 'text_' + self.split_type + multifile_option + '_format.' + filetype)
        self.docInfo_filename = os.path.join(self.output_dir, 'text_' + self.split_type + multifile_option + '_doc_info.' + filetype)
        self.ttype_filename = os.path.join(self.output_dir, 'text_' + self.split_type + multifile_option + '_ttype.' + filetype)
        if self.consolidate_page_info:
            self.pageInfo_filename = os.path.join(self.output_dir, 'text_' + self.split_type + self.pageInfo_suffix +'.' + filetype) 
            self.corrections_filename = os.path.join(self.output_dir, 'corrections.' + filetype)
            self.orphans_filename = os.path.join(self.output_dir,'orphans.' + filetype)
        else:
            self.pageInfo_filename = os.path.join(self.output_dir, 'text_' + self.split_type  + multifile_option  + self.pageInfo_suffix +'.'+ filetype) 
            self.corrections_filename = os.path.join(self.output_dir, 'corrections' +  multifile_option + "." + filetype)
            self.orphans_filename = os.path.join(self.output_dir,'orphans' +  multifile_option + "." + filetype)
            
    
    def appendToList(self, dataset_list, response_list):
        dataset_list['words'].extend(response_list['words'])
        #dataset_list['bbox'].extend(response_list['bbox']) 
        dataset_list['pages'].extend(response_list['pages']) 
        dataset_list['raw_text'].update(response_list['raw_text']) 
        if response_list.get('corrections') is not None and  len(response_list['corrections']) >0:
            dataset_list['corrections'].extend(response_list['corrections']) 
            
        if response_list.get('orphans') is not None and  len(response_list['orphans']) >0:
            dataset_list['orphans'].extend(response_list['orphans'])  
            
        return dataset_list

    def prepare_distribution(self, distrib_tool, dataset_batch, total_workers):
        print('Distribute work of ' + str(len(dataset_batch)) + ' batches. Saving files by', self.multifile_option)
                    
        start_time = time.time() 
        temporal_file = None
        if self.multifile_option == "batch":
            temporal_file = os.path.join(self.output_dir, self.get_temp_doclist_name())
            current_documents = [{'file_name':Path(item['datafile']).stem} for mini_batch in dataset_batch for item in mini_batch['data']]
            with open(temporal_file,"w") as f:
                json.dump(current_documents, f)
            
        results = distribute(distrib_tool, dataset_batch, apply_ocr_and_html, total_workers)

        end_time = time.time() - start_time
        
        print('Collecting results from ' + str(len(dataset_batch)) + ' batches. Total processing time: ' + str(end_time/60) + ' minutes.')
        
        multifile_index = self.get_last_batch_number(self.output_dir)
        consolidated_results = {}
        dataset_list = {}
        records_per_file = {}
        error_list = []
        for i_result, o_result in enumerate(results):
            document_list =  o_result[0]
            raised_error = o_result[1]
            if raised_error is not None:
                error_list.append(raised_error)

            for doc_result in document_list:        
                if len(document_list[doc_result]['words']) >0:
                    doc_name = Path(doc_result).stem
                    if records_per_file.get(doc_name) is None: records_per_file[doc_name] = {'total':0,'last_file':None}
                    records_per_file[doc_name]['total'] += 1
                    records_per_file[doc_name]['last_file'] = doc_result

            for i_doc, doc_result in enumerate(document_list):
                if len(document_list[doc_result]['words']) >0:
                    if len(document_list[doc_result]['pages']) >0: 
                        doc_name = Path(doc_result).stem
                        if consolidated_results.get(doc_name) is None:
                            consolidated_results[doc_name] = document_list[doc_result]
                        else:
                            for result_key in document_list[doc_result]:
                                if type(consolidated_results[doc_name][result_key]) ==list:
                                    consolidated_results[doc_name][result_key].extend(document_list[doc_result][result_key])
                                elif type(consolidated_results[doc_name][result_key]) ==dict:
                                    consolidated_results[doc_name][result_key].update(document_list[doc_result][result_key])
                                else:
                                    consolidated_results[doc_name][result_key] = document_list[doc_result][result_key]                  
                                        
                        #assert len(consolidated_results[doc_name]['words']) == len(consolidated_results[doc_name]['bbox'])                   

                        if self.multifile_option == "document" or self.multifile_option == "page":
                            if records_per_file[doc_name]['last_file'] == doc_result:
                                self.saveFiles(consolidated_results[doc_name], multifile_index, self.multifile_option)
                                consolidated_results = {} 
                        else:
                            if dataset_list.get('pages') is None: 
                                dataset_list = consolidated_results[doc_name]
                            else:
                                for result_key in consolidated_results[doc_name]:
                                    if type(consolidated_results[doc_name][result_key]) ==list:
                                        dataset_list[result_key].extend(document_list[doc_result][result_key])
                                    if type(consolidated_results[doc_name][result_key]) ==dict:
                                        dataset_list[result_key].update(document_list[doc_result][result_key])
                    else:
                        print('ERROR No readable results. ' , list(document_list.keys()))

        #save files
        if dataset_list.get('pages') is not None:
            if len(dataset_list['pages'])>0 and self.multifile_option != "document" and self.multifile_option != "page":
                self.saveFiles(dataset_list, multifile_index, multifile_option = self.multifile_option)
                dataset_list = {}
        
        #deleting temporal file
        if temporal_file is not None:
            os.remove(temporal_file)

        if len(error_list) > 0 and len(error_list[0])>0:
            output_file_errors = os.path.join(self.output_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '_errors_docs.json')
            with open(output_file_errors, 'w') as file:    
                json.dump(error_list, file)
                print('Saving errors file',output_file_errors)

    def saveFiles(self, dataset_list, multifile_index, multifile_option='batch'):
        is_saved = False
        backup_files = []
        if multifile_option=='page':
            pages = [i for i in range(len(dataset_list['pages']))]
        else:
            pages = [None]

        document_name = None
        if multifile_option =='document' or multifile_option == 'page':
            document_name = dataset_list['pages'][0]['file_name']  
        
        is_saved = True

        for page in pages: 
            if multifile_option == 'page':
                document_name = dataset_list['pages'][page]['file_name'] + "." + str(dataset_list['pages'][page]['page_number']) 
            
            #saving textual info
            if len(dataset_list['words'])>0 and is_saved:
                is_saved = False
                current_pages = {item['page_number']:0 for item in dataset_list['words']}
                if page is None: 
                    list_to_save = dataset_list['words']
                    raw_text_list = dataset_list['raw_text']
                    raw_corrected_text_list = dataset_list['raw_corrected_text']
                else:
                    list_to_save = [item for item in  dataset_list['words'] if item['page_number']==(page+1)]
                    raw_text_list =  {item: dataset_list['raw_text'][item] for item in  dataset_list['raw_text'] if int(item.split("__")[-1])==(page+1)}
                    raw_corrected_text_list = {'pages':[item] for i_page, item in enumerate(dataset_list['raw_corrected_text']['pages']) if i_page ==page}

                if len(raw_text_list)>0:
                    #saving words and text info
                    is_saved, backup_file =save_data(list_to_save, self.text_filename, multifile_index, multifile_option= multifile_option, document_name=document_name)
                    backup_files.append(backup_file)
                    if not is_saved:
                        restore_delete_backup_data(backup_files, is_restore=True)
                    
                    #saving raw_text
                    is_saved, backup_file =save_data(raw_text_list, self.raw_text_filename, multifile_index, multifile_option= multifile_option, document_name=document_name, raw_file=True)
                    backup_files.append(backup_file)
                    if not is_saved:
                        restore_delete_backup_data(backup_files, is_restore=True)
                    
                    #saving corrected raw_text
                    is_saved, backup_file =save_data(raw_corrected_text_list, self.raw_text_corrected_filename, multifile_index, multifile_option= multifile_option, document_name=document_name, raw_file=True)
                    backup_files.append(backup_file)
                    if not is_saved:
                        restore_delete_backup_data(backup_files, is_restore=True)

            #saving bbox files
            '''
            if len(dataset_list['bbox'])>0 and is_saved:
                list_to_save =  dataset_list['bbox'] if page is None else [item for item in  dataset_list['bbox'] if item['page_number']==(page+1)] 
                is_saved, backup_file =save_data(list_to_save, self.bbox_filename, multifile_index, multifile_option= multifile_option, document_name=document_name)
                backup_files.append(backup_file)
                if not is_saved:
                    restore_delete_backup_data(backup_files, is_restore=True)

            '''
            
        document_name = dataset_list['pages'][0]['file_name']  
        
        #saving page info            
        if len(dataset_list['pages']) >0 and is_saved:
            is_saved, backup_file = save_data(dataset_list['pages'], self.pageInfo_filename, multifile_index, multifile_option= multifile_option, document_name=document_name)
            backup_files.append(backup_file)
            if not is_saved:
                restore_delete_backup_data(backup_files, is_restore=True)
        
        #saving correction files
        if len(dataset_list['corrections']) >0 and is_saved: 
            is_saved, backup_file =save_data(dataset_list['corrections'], self.corrections_filename, multifile_index, multifile_option= multifile_option, document_name=document_name)
            backup_files.append(backup_file)        
            if not is_saved:
                restore_delete_backup_data(backup_files, is_restore=True)
        
        #saving orphans files
        if len(dataset_list['orphans']) >0 and is_saved:
            is_saved, backup_file =save_data(dataset_list['orphans'], self.orphans_filename, multifile_index, multifile_option= multifile_option, document_name=document_name)
            backup_files.append(backup_file)        
            if not is_saved:
                restore_delete_backup_data(backup_files, is_restore=True)                


        if is_saved:
            restore_delete_backup_data(backup_files, is_restore=False)
    
    def generateSingleFile(self, file_path, detect_lang, lang, max_pages_per_time=None):
        total_time_min = 0
        self.detect_lang = detect_lang
        self.lang = lang  

        start_time = time.time()
    
        error_list = []
        
        if os.path.exists(file_path):
            document_path =  file_path
            is_downloaded = True
            is_link = False
        else:
            is_link = True
            destination_dir = os.path.join(self.output_dir,"temp")  
            file_name = re.findall(r'\w+', file_path.split("/")[-1])
            file_name = ' '.join(file_name)
            file_destination_temp_path = os.path.join(destination_dir, file_name + ".pdf")
            is_downloaded = getFileFromLink(file_path, file_destination_temp_path)
            if is_downloaded:
                document_path = file_destination_temp_path
        
        if is_downloaded:
            
            with pikepdf.open(document_path) as pdf:
                total_pages =  len(pdf.pages)
                if max_pages_per_time is None:
                    total_groups  = 1
                    max_pages_per_time = total_pages
                else:
                    total_groups = int(math.ceil(total_pages/max_pages_per_time))

            info = {}
            result = {}
            for i_group in range(total_groups):
                info['data'] = [{'datafile':document_path, 'split_type':self.split_type, 'detect_lang': self.detect_lang, 'lang':self.lang,'page_start':i_group*max_pages_per_time,'page_end': (i_group+1)*max_pages_per_time}]
                info['modules'] = self.modules
                        
                result_run, raised_error = apply_ocr_and_html(info)
                
                if raised_error is not None and len(raised_error)>0:
                    error_list.append(raised_error)

                result_run = [result_run[item] for item in result_run]
                if i_group==0:
                    result = result_run
                else:
                    result[0]['words'].extend(result_run[0]['words'])
                    #result[0]['bbox'].extend(result_run[0]['bbox'])
                    result[0]['pages'].extend(result_run[0]['pages'])
                    result[0]['raw_text'].update(result_run[0]['raw_text'])
                    result[0]['raw_corrected_text']['pages'].extend(result_run[0]['raw_corrected_text']['pages'])
                    result[0]['raw_corrected_text']['number_pages'] = len(result[0]['raw_corrected_text']['pages'])
                    
                    if len(result_run[0]['corrections']) >0: result[0]['corrections'].extend(result_run[0]['corrections'])
                    if len(result_run[0]['orphans']) >0: result[0]['orphans'].extend(result_run[0]['orphans'])

            if os.path.exists(document_path) and is_link: os.remove(document_path)
            
            if len(result)>0 and len(result[0]['pages']) >0: 
                self.saveFiles(result[0], multifile_index = None, multifile_option = self.multifile_option) 
            else:
                print('ERROR No readable results. ' , file_path)

            end_time = time.time() 
            total_time = end_time - start_time           
            total_time_min =  total_time/60
        else:
            print('Error downloading link',file_path)
        
        
        if len(error_list) > 0 and len(error_list[0])>0:
            output_file_errors = os.path.join(self.output_dir, datetime.now().strftime("%Y%m%d%H%M%S") + '_errors_docs.json')
            with open(output_file_errors, 'w') as file:    
                json.dump(error_list, file)
                print('Saving errors file',output_file_errors)
                
        return is_link, is_downloaded, total_time_min

        
    def pre_process_list(self, link_list):
        list_sized_docs = [[0,'', item] for item in link_list] 
        broken_links = []
        
        for i_doc,doc_info in enumerate(list_sized_docs):
            destination_dir = os.path.join(args.output_dir,"temp") 
            file_name = doc_info[-1].split("/")[-1]
            file_destination_temp_path = os.path.join(destination_dir, file_name + ".pdf")
            is_downloaded = getFileFromLink(doc_info[-1], file_destination_temp_path)
            if is_downloaded: 
                with fitz.open(file_destination_temp_path) as doc_fitz_temp:            
                    try:
                        total_pages = len(doc_fitz_temp) 
                        list_sized_docs[i_doc][0] = total_pages
                    except:
                        print("ERROR! Can't open document", file_destination_temp_path) 
                        list_sized_docs[i_doc][0] = None
                        broken_links.append(file_destination_temp_path)
                list_sized_docs[i_doc][1] = file_destination_temp_path 
            else:
                list_sized_docs[i_doc][1] = None

        list_sized_docs = [item for item in list_sized_docs if item[1] is not None]
        list_sized_docs.sort(reverse=True)
        print('List of total pages per document',[item[0] for item in list_sized_docs])
        list_sized_docs = [[item[0],item[1]] for item in list_sized_docs] 
        return list_sized_docs, broken_links

    def generateFiles(self, detect_lang, lang, worker_load, total_workers, max_docs_per_run, max_pages_per_run,distrib_tool, from_links=False, min_info=False): 
        
        self.detect_lang = detect_lang
        self.lang = lang
        self.worker_load = worker_load
        self.total_workers = total_workers
        

        if from_links:
            if self.filter_files is None:
                raise Exception('A file with document\'s links should be specified in the argument filter_list_path')    
        else:
            total_file_list = exploreDirectory(self.data_dir,filtered_dict=self.loaded_docs)
        
            if self.filter_files is None:
                print('Starting generation process. Documents in directory:' + str(len(total_file_list)))
            else:
                print('Starting generation process. Documents in directory:' + str(len(total_file_list)), ". Documents in Filter file: ", len(self.filter_files))

        dataset_batch = []         
        loaded_docs = self.loaded_docs
        
        #only docs in scope
        if from_links:
            print('Number links in file', len(self.filter_files))
            refined_file_list = [item for item in self.filter_files if loaded_docs.get(item.split("/")[-1]) is None]
            print('Number links in scope and not processed', len(refined_file_list))
        else:
            refined_file_list = [item for item in total_file_list if self.filter_files.get(item.stem) is not None]
            print('Number downloaded documents in scope', len(refined_file_list))
            #only not loaded docs
            refined_file_list = [item for item in refined_file_list if loaded_docs.get(item.stem) is None]
            print('Number downloaded documents in scope and not processed', len(refined_file_list))


        max_iteration = min(max_docs_per_run, len(refined_file_list))
        total_times = []

        working_docs = worker_load * total_workers
        for i_range in range(0,max_iteration,working_docs): #tqdm.tqdm(
            number_docs = 0
            start_time = time.time()
            list_sized_docs, broken_links = self.pre_process_list(refined_file_list[i_range:i_range+working_docs])
            avg_page_load = math.ceil(sum([item[0] for item in list_sized_docs])/total_workers)

            batch_of_workers = [{'pages':0,'data':[]} for item in range(total_workers)]
            current_worker = 0
            for o_document in list_sized_docs:
                number_docs +=1
                if batch_of_workers[current_worker]['pages'] > avg_page_load:
                    if current_worker +1 < total_workers:
                        current_worker += 1

                number_segments = math.ceil(o_document[0]/max_pages_per_run) 
                for i in range(number_segments):
                    page_end=(i+1)*max_pages_per_run-1
                    if page_end > o_document[0]:
                        page_end = o_document[0]-1
                    data_info = {'datafile':o_document[1], 'split_type':self.split_type, 'detect_lang': self.detect_lang, 'lang':self.lang,'page_start':i*max_pages_per_run,'page_end': page_end}
                    batch_of_workers[current_worker]['data'].append(data_info)
                batch_of_workers[current_worker]['pages'] += o_document[0] 
                current_worker +=1

                if current_worker >= total_workers:
                    current_worker = 0
                
            dataset_batch= [{'data': item['data'], 'modules': self.modules} for item in batch_of_workers if len(item)>0]
            self.total_workers = len(dataset_batch)
            print('Number of processing workers:', self.total_workers, "Total documents:", number_docs)
            for i_w, worker_load in enumerate(batch_of_workers):
                sLoad = '; '.join([Path(data['datafile']).stem + ':' + str(data['page_end'] - data['page_start']+1) for data in worker_load['data']])  
                print('Workload per worker (',i_w,'-',worker_load['pages'],')', sLoad)            
            
            self.prepare_distribution(distrib_tool, dataset_batch, self.total_workers)
            
            end_time = time.time() - start_time
            total_min = end_time/60
            total_times.append({'time':total_min, 'total_docs':number_docs, 'total_pages': [item['pages'] for item in batch_of_workers]})
            if from_links:
                for o_document in list_sized_docs:
                    if os.path.exists(o_document[1]): os.remove(o_document[1])

            loaded_docs = self.get_loaded_and_working_docs(print_info=False)
            print('Distribution of pages by worker:', total_times[-1]['total_pages'] , '. Total time:', total_times[-1]['time'], 'min')
    
        if len(broken_links)>0:
            with open(os.path.join(self.output_dir,'broken_links.txt')) as f:
                f.writelines(broken_links)
        return total_times
        
    def get_temp_doclist_name(self):
        oDate = datetime.now()
        return oDate.strftime("%Y_%m_%d_%H_%M_%S_%f") + ".docs_tmp" 

    def generate_document_metadata(self, data_dir, output_dir, input_file, 
        required_indexes = {'company_id':'company_id','company_industry':'company_industry','file_name':'file_name','document_type':'document_type','document_year':'document_year'}):
        with open(os.path.join(data_dir, input_file),'r', encoding='utf-8') as metadata_file:
            metadata_text_list = json.load(metadata_file)
            metadata_headers_extra = [item for item in metadata_text_list[0].keys() if required_indexes.get(item) is None]
            metadata_text_dict = {item['file_name']:item for item in metadata_text_list}
        
        with open(os.path.join(data_dir,self.pageInfo_filename),'r') as page_info:
            page_info = page_info.readlines()
            indexed_docs = [item.split('\t') for item in page_info[1:]]
            indexed_docs = {i:(True,metadata_text_dict[item[0]], item[1]) if metadata_text_dict.get(item[0]) is not None else (False,item, item[1]) for i, item in enumerate(indexed_docs)}
        file_name =  os.path.join(output_dir, self.docInfo_filename) 
        document_info_header = 'file_name\tpage_number\tcompany_id\tdocument_year\tdocument_type\tcompany_industry'
        if len(metadata_headers_extra) >0:
            document_info_header+='\t' + '\t'.join(metadata_headers_extra)
        data_list = []

        for i in indexed_docs:
            if indexed_docs[i][0]:
                data_list.append(indexed_docs[i][1][required_indexes['file_name']]+ '\t' + indexed_docs[i][2] + '\t' + 
                    str(indexed_docs[i][1][required_indexes['company_id']]) + '\t' + str(indexed_docs[i][1][required_indexes['document_year']]) + '\t' + 
                    indexed_docs[i][1][required_indexes['document_type']]+ '\t' + indexed_docs[i][1][required_indexes['company_industry']] + '\t' +
                    '\t'.join([indexed_docs[i][1][item].replace('\t',' ') if indexed_docs[i][1][item] is not None else ''  for item in metadata_headers_extra]))
            else:
                data_list.append(indexed_docs[i][1][0]+ '\t'+indexed_docs[i][2] + '\t' + '0' + '\t' + '0' + '\t' + '\t'+ ''.join(['\t' for item in metadata_headers_extra]))
            
        save_data(data_list, file_name, document_info_header)    

def read_data(file_name):
    oFileName = Path(file_name)
    _data= []
    if oFileName.suffix == '.txt':
        try:
            with open(oFileName, encoding='utf-8') as f:
                _data = f.readlines() 
        except UnicodeDecodeError:
            _data = []
            with open(oFileName) as f:
                temp_data = f.readlines() 
            headers = []
            for i_row, row in enumerate(temp_data):
                if i_row ==0:
                    headers = [item for item in row.replace("\n","").split(";") if len(item) >0]
                else:
                    _data.append({header:row.split(";")[i_h] for i_h, header in enumerate(headers)})
    
    elif oFileName.suffix == '.json':
        try:
            with open(oFileName, 'r') as f:
                _data = json.load(f) 
        except Exception as ex:
            print('ERROR while openining file', oFileName.name)
            print(ex)
    return _data
            
def save_data(data_list, file_name, multifile_index=-1, filetype='json', max_retries=3, multifile_option='batch', document_name=None, raw_file = None):
    oFile = Path(file_name)
    #document_name_stem = ''.join(re.findall(r'\w+', oFile.stem))
    document_name_stem = oFile.stem
    if multifile_option is None or multifile_option == 'batch':
        if multifile_index is not None:
            suffix = "_bth" + str(multifile_index+1)
            file_name = os.path.join(oFile.parent, document_name_stem + suffix + oFile.suffix)

        file_name_bck = file_name + ".bck" 
        file_exists = False

        if os.path.exists(file_name):
            file_exists = True
        else:
            file_name_bck = None

        if filetype =='json': 
            existing_data = []
            if file_exists:      
                with open(file_name, 'r') as file:    
                    existing_data = json.load(file) 

                os.rename(file_name,file_name_bck)
                print('Creating backup file ', file_name_bck)
                

            existing_data.extend(data_list)

            #save and validate
            is_saved = saveJsonRetries(file_name, existing_data, max_retries)
            if is_saved is not None and not is_saved:
                return False, file_name_bck

        elif filetype =='txt': #backup
            with open(file_name, 'a', encoding='utf-8') as file:  
                if not file_exists:     
                    header = '\t'.join([item for item in data_list[0]])
                    file.write(header + '\n') 
                for row in data_list: 
                    row_content = '\t'.join([str(row[item]) for item in row]) + '\n'
                    file.write(row_content)
                print('updating file ', file_name)
    else:
        print('Saving data by ', multifile_option, 'with name', document_name)
        file_name_bck = None 
        
        if multifile_option == 'page' or multifile_option=='document':
            if document_name is not None:
                file_name = os.path.join(oFile.parent, document_name_stem.replace("#" + multifile_option, document_name) + str(oFile.suffix))
        
        if filetype =='json' or raw_file: 
            #save and validate
            is_saved = saveJsonRetries(file_name, data_list, max_retries)
            if is_saved is not None and not is_saved:
                return False, None

        elif filetype =='txt' and not raw_file:
            with open(file_name, 'w', encoding='utf-8') as file:   
                header = '\t'.join([item for item in data_list[0]])
                file.write(header + '\n')
                for row in data_list:
                    row_content = '\t'.join([str(row[item]) for item in row]) + '\n'
                    file.write(row_content)
                print('registering file ', file_name)
        
    return True, file_name_bck

def saveJsonRetries(file_name, existing_data, max_retries):
    retry_number = 0
    while(retry_number<max_retries):
        with open(file_name, 'w') as file:    
            json.dump(existing_data, file)
            print('updating file ', file_name)
        try:
            with open(file_name, 'r') as file:    
                temp_data = json.load(file)
            retry_number = max_retries + 1
            temp_data = None
        except: 
            retry_number += 1
        if retry_number ==max_retries:
            print('File not saved. Error while saving, recovering last file')
            return False

def restore_delete_backup_data(list_backup_files, is_restore):
    for backup_file in list_backup_files:
        if backup_file is not None:
            current_file = backup_file.split(".bck")[0]
            if is_restore:
                os.remove(current_file) #is corrupted
                os.rename(backup_file, current_file)
                print('Recoving backup file', current_file) 
            else:
                os.remove(backup_file)
                print('Backup file removed', backup_file)


class Object(object):
    pass    
    
class FDExt(Dataset):

    def __init__(self, data_dir, output_dir, action=None, total_records=None, dataset_name=None): 
        self.train = Object()
        self.train.text = []
        self.train.all_input_ids = []
        self.train.all_input_mask = []
        self.train.all_segment_ids = []
        self.train.all_label_ids = []
        self.train.labels2idx = {}
        self.train.idx2labels = {}
        self.train.weights = {}
        self.train.weight_strategy = None
        self.is_train = True if action is not None and 'train' in action else False  
        self.special_tokens = self.get_special_tokens()
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.total_records = total_records
        self.dataset_name = None
        
        if dataset_name is None:
            self.dataset_name = self.get_dataset_name()
        else:
            self.dataset_name = dataset_name
        
        print('Dataset name:',self.dataset_name)

        self.args_ret = {'is_bert_lstm': False, 'lstm_sequence_length': 0, 'lstm_stacked_layers':0}

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
 
    def get_special_tokens(self, as_vector = False):
        special_tokens = {'number': "[NUMBER]", 'date': "[DATE]", 'phone': "[PHONE]",'percentage':"[PERC]",'code':"[CODE]", 'positive': "[POSITIVE]", 'negative': "[NEGATIVE]","small":"[SMALL]","big":"[BIG]"}
        sizes_log = {'small':'SMALL','big':'BIG'}
        for exp_log in range(1, 10):
            for size_log in sizes_log:
                special_tokens[size_log + "_number" + str(exp_log)] ="["+ sizes_log[size_log] + "_NUMBER_" + str(str(exp_log))+"]"
        
        if not as_vector:
            return special_tokens
        else:
            return [special_tokens[item] for item in special_tokens]


    def loadDataset(self, tasks_= 'default', filter_last_doc=None, filter_type_doc=None, filter_industry_cia=None , filter_lang=None, load_level = 'Default', perc_data = 1, additional_filters = None, labels_path=None, data_sel_position='first', max_number_files=None):
        '''
        tasks_: Data to return togueter with the text
        doc_filter: If there is a document metadata file, a set of filters can be specified in a dictionary that could have the following keys:
        - last_doc: -1: referes to last year, -3 refers to the last 3 years, None refers to no filter.
        - type_doc: Only documents with the specified value (document type)
        - cia_ind: Only documents that belongs to the specified value (industry)
        - additional_filters: Other columns as filters under a dictionary structure
        - perc_data: Percentaje of the dataset to load
        - data_sel_position: If the data is going to be taken from the begining or the end of the dataset        
        '''
        
        if self.load_dataset_from_file(): 
            return False
            
        doc_filters = {}

        if filter_last_doc is not None:
            doc_filters['last_doc'] = filter_last_doc
        if filter_type_doc is not None: 
            doc_filters['type_doc'] =  filter_type_doc
        if filter_industry_cia is not None: 
            doc_filters['ind_cia'] =  filter_industry_cia
        if filter_lang is not None: 
            doc_filters['lang'] =   filter_lang

        if additional_filters is not None:
            for filter_ in additional_filters:
                doc_filters[filter_] = additional_filters[filter_]

        if not os.path.exists(self.data_dir):
            raise Exception('The data directory does not exists. ')
        
        print('Load level: ' + load_level)
        if labels_path is not None and not os.path.exists(labels_path): raise Exception("Labels' file doesn't exists")
        data = {}
        base_files = {}        
        #Loading set of files into independent lists
        current_number_files = 0
        for file in os.listdir(self.data_dir):
                oFile = Path(file) 
                file_composition = oFile.stem.split("_")
                sFileName = os.path.join(self.data_dir, oFile.name)
                if 'text_' == oFile.name[:5] and  (oFile.suffix == '.txt' or oFile.suffix == ".json"):
                    print(oFile.stem)
                    '''
                    if len(file_composition)>=3 and file_composition[2]=="bbox" and not (load_level=='page' or 'text_min' in load_level):
                        if 'bbox' in tasks_ :
                            if data.get('bbox') is None:    
                                data['bbox'] = read_data(sFileName)
                            else:
                                temp_text = read_data(sFileName) 
                                data['bbox'].extend(temp_text) 
                            print('>>>bbox')
                    el
                    '''
                    if len(file_composition)>=3 and file_composition[2]=="format" and not (load_level=='page' or 'text_min' in load_level):
                        if 'format' in tasks_:  
                            if data.get('format') is None:    
                                data['format'] = read_data(sFileName)
                            else:
                                temp_text = read_data(sFileName) 
                                data['format'].extend(temp_text)  
                            print('>>>>format')                                
                    elif 'page' in oFile.stem:  
                        if  file_composition[3] =='info' and (len(file_composition)==4 or (len(file_composition)==5 and file_composition[4][:3]=="bth")):
                            if  data.get('page_info') is None:
                                data['page_info'] = read_data(sFileName)
                            else:
                                 data['page_info'].extend(read_data(sFileName))
                            print('>>>page_info') 
                        else:
                            base_files[oFile.stem] = 'page_info'
                            data[oFile.stem] = read_data(sFileName)
                            print('>>>' + oFile.stem)

                    elif 'doc' in oFile.stem and not (load_level=='page'):  
                        if  '_doc'== oFile.stem[-4:]:
                            data['doc_info'] = read_data(sFileName)
                            print('>>>doc_info') 
                        else:
                            base_files[oFile.stem] = 'doc_info'
                            data[oFile.stem] = read_data(sFileName)
                            print('>>>' + oFile.stem)
                    elif 'cia' in oFile.stem and not (load_level=='page'):
                        if  '_cia'== oFile.stem[-4:]:
                            data['cia_info'] = read_data(sFileName)
                            print('>>>cia_info') 
                        else:
                            base_files[oFile.stem] = 'cia_info'
                            data[oFile.stem] = read_data(sFileName)
                            print('>>>' + oFile.stem)
                    elif 'text_' in oFile.stem[:5] and load_level !='page'  and (len(oFile.stem.split("_"))==2 or len(oFile.stem.split("_"))==3 )and 'doc' not in oFile.stem and 'cia' not in oFile.stem and 'page' not in oFile.stem: 
                        current_number_files += 1
                        if max_number_files is None or current_number_files < max_number_files:
                            if data.get('text') is None:    
                                data['text']  = read_data(sFileName)
                            else:
                                temp_text = read_data(sFileName) 
                                data['text'].extend(temp_text) 
                            
                            print('>>>>text')  
                        else:
                            print('Ignored') 
                        
                            
                        
        #get name of files which are extra to the base files
        base_files_root = {base_files[item]:0 for item in base_files}
        extra_files = [item for item in list(data.keys()) if item not in ['text','cia_info','page_info','doc_info']]
        
        for extra_file in extra_files:
            #extracting information from extra files
            current_record = ''
            is_page = True
            doc_dict = {}
            column = 'company_id' if 'cia' in extra_file else 'file_name'
            if type(data[extra_file][0]) == dict:
                is_page = False if ('doc' in extra_file or 'cia' in extra_file) else True
                dict_temp = {}
                for record_line_tmp in len(data[extra_file]): 
                    if not is_page:
                        dict_temp[record_line_tmp[column]] = record_line_tmp
                    else:
                        dict_temp[record_line_tmp['file_name'] + "-" + str(record_line_tmp['page_number'])] = record_line_tmp
                        doc_dict[record_line_tmp['file_name']] = 1
            else:
                headers_items = [item for item in data[extra_file][0][:-1].split('\t')]
                if 'doc' in extra_file or  'cia' in extra_file:
                    is_page = False
                    i_start = 0
                else:
                    i_start = 1
                empty_record = {item:None for i,item in enumerate(headers_items) if i>i_start} 

                file_header_str = '\t'.join([item for item in empty_record])
                
                dict_temp = {}
                for record_line_tmp in range(1,len(data[extra_file])): 
                    data_line_temp = data[extra_file][record_line_tmp][:-1].split('\t')
                    data_item = {headers_items[i]:data_line_temp[i] for i in range(len(headers_items)) if i > i_start}
                    
                    if not is_page:
                        dict_temp[str(data_line_temp[0])] = data_item #'\t'.join(data_line_temp[1:])
                    else:
                        dict_temp[str(data_line_temp[0]) + "-" + str(data_line_temp[1])] = data_item #'\t'.join(data_line_temp[2:])
                        doc_dict[str(data_line_temp[0])] = 1
                
            #integrating extra information into a base list
            if data.get(base_files[extra_file]) is None:
                data[base_files[extra_file]] = copy.deepcopy(data[extra_file])
            else:
                if type(data[base_files[extra_file]][0]) == dict:
                    for i_record,record_line in enumerate(data[base_files[extra_file]]):
                        record_key = record_line[column]
                        if is_page:
                            record_Key2 = record_line['page_number']
                            if doc_dict.get(str(record_key)) is not None:
                                current_record = dict_temp.get(str(record_key) + "-" + str(record_Key2)) 
                            else:
                                current_record = None 
                        else:
                            current_record = dict_temp.get(str(record_key)) 

                        if current_record is not None: 
                            data[base_files[extra_file]][i_record].update(current_record) 
                        else:
                            data[base_files[extra_file]][i_record].update(empty_record)
                else:
                    for i_record,record_line in enumerate(data[base_files[extra_file]]):
                        record_id = record_line.split('\t')[0]
                        if is_page:
                            record_id2 = record_line.split('\t')[1]
                        if i_record==0:
                            data[base_files[extra_file]][i_record] = record_line[:-1]+'\t'+file_header_str+'\n'
                        else:
                            if is_page:
                                if doc_dict.get(str(record_id)) is not None:
                                    current_record = dict_temp.get(str(record_id) + "-" + str(record_id2)) 
                                else:
                                    current_record = None 
                            else:
                                current_record = dict_temp.get(str(record_id)) 

                            if current_record is not None:
                                current_record_str = '\t'.join([current_record[item] for item in current_record])
                                data[base_files[extra_file]][i_record] = record_line[:-1]+'\t'+current_record_str+'\n'
                            else:
                                empty_record_str = '\t'.join(['' for item in empty_record])
                                data[base_files[extra_file]][i_record] = record_line[:-1]+'\t'+ empty_record_str+'\n'
            
            data[extra_file] = None

        if data.get('text') is None and load_level !='page': raise Exception('Text file not found.')
        #if data.get('bbox') is not None:
        #    assert len(data['text']) == len(data['bbox'])
        if data.get('format') is not None:
            assert len(data['text']) ==len(data['format'])
        
        if labels_path is not None:
            self.loaded_labels = read_data(labels_path)
            print('>>>>labels', len(self.loaded_labels), 'from ' , labels_path) 
        else:
            self.loaded_labels = None
        #Reducing the dataset
        if perc_data < 1: 
            initial_rows = len(data['text'])
            new_rows = int(perc_data*initial_rows)
            print('Having the ', data_sel_position , 'part of the dataset')
            if data_sel_position == 'first':
                data['text'] = data['text'][:new_rows] 
            else:
                header = data['text'][0]
                data['text'] = data['text'][-new_rows:]
                data['text'].insert(0, header)

            #removing the last/first document to avoid to work with incomplete documents
            if initial_rows>0:
                if type(data['text'][0])==dict:
                    if data_sel_position == 'first':
                        document_name = data['text'][-1]['file_name']
                    else:
                        document_name = data['text'][0]['file_name']
                    data['text'] = [item for item in  data['text'] if item['file_name']!=document_name]
                else:
                    if data_sel_position == 'first':
                        last_row = data['text'][-1].split('\t')
                    else:
                        last_row = data['text'][1].split('\t')
                    if len(last_row) > 0:
                        document_name = last_row[0]
                        data['text'] = [item for item in  data['text'] if item[:len(document_name)]!=document_name]
            
            print('Reduced from : ', initial_rows, 'rows, to: ' , new_rows, 'rows', "~",perc_data*100, "% of data")

        #Loading row info    
        dataset = {}   
        if load_level != 'page':
            dataset = self.create_dict_from_datatext(data['text'], types_list=[str, int, int, str, int, int, int, int, float, float, float], debugnote='not page, text') 
            self.train.text = [item['text'] for item in dataset] 


        self.number_documents = len({item['file_name']:0 for item in dataset})
        print('Initial reduced dataset with ', self.number_documents, ' documents', 'and', len(dataset), "records")
        self.page_info = None

        if dataset[0].get('page_number') is not None:
            number_pages = len({item['file_name']+"-"+str(item['page_number']):0 for item in dataset})
            print('Initial dataset with ', number_pages, 'pages')
        
        if not load_level =='text_min' and  data.get('page_info') is not None:
            print('working with page info')
            print(data['page_info'][0])
            dataset_pages = self.create_dict_from_datatext(data['page_info'], types_list=[str, int, float, float, str, int, str, float, str, float, str, int], debugnote='page info') 
            data['page_info'] = None
            dataset = self.merge_dataset(dataset, dataset_pages, level='page')


            if not 'text_min' == load_level:
                self.page_info = dataset_pages
    
            del dataset_pages

        if not 'text_min' in load_level:        
            '''
            if data.get('bbox') is not None:
                print('working with bbox')
                dataset_tmp = self.create_dict_from_datatext(data['bbox'], types_list=[str, int, int, str, float, float, float], debugnote='bbox') 
                dataset = self.merge_dataset(dataset, dataset_tmp, level='record')
                data['bbox'] = None
                del dataset_tmp
            '''
            
            if data.get('format') is not None:
                print('working with format')
                dataset_tmp = self.create_dict_from_datatext(data['format'], types_list=[str, int, int, str], debugnote='format') 
                data['format'] = None
                dataset = self.merge_dataset(dataset, dataset_tmp, level='record')
                del dataset_tmp

        if data.get('doc_info') is not None:
            print('working with doc info')
            if dataset is None: dataset = {}
            dataset_docs = self.create_dict_from_datatext(data['doc_info'], types_list=[str, int, int], debugnote='doc_info')  
            dataset = self.merge_dataset(dataset, dataset_docs, level='doc') 
            
            if self.page_info is not None:
                self.page_info =  self.merge_dataset(self.page_info , dataset_docs, level='doc')
            else:
                if 'text_min' not in load_level:
                    self.page_info = dataset_docs 
            del dataset_docs

        if data.get('cia_info') is not None: 
            if dataset is None: dataset = {}
            dataset_docs = self.create_dict_from_datatext(data['cia_info'], types_list=[str],debugnote='cia_info')

            dataset = self.merge_dataset(dataset, dataset_docs, level='cia')
            del dataset_docs
            

        print('Previous filtering', Counter([item.get('risk_desc') for item in dataset if item.get('risk_desc') is not None]))
        print('Number companies', len({item.get('company_id') for item in dataset}))
        print('Number documents', len({item.get('file_name') for item in dataset}))
        if data.get('doc_info') is not None or data.get('page_info') is not None: 
            self.dataset = dataset
            dataset = self.filter_dataset(doc_filters)      
            self.dataset = dataset
            data['cia_info'] = None
            data['doc_info'] = None
        
        if 'lang' in load_level or load_level=='Default':
            if  'text_min' in load_level:
                self.train.text = []
                self.train.text_lang = [{'file_name':item['file_name'], 'page_number':item['page_number'], 'line_number': item['line_number'], 'language': item['language'],'text':item['text']} for item in dataset]        
                del dataset
            else:
                self.train.text_lang = [{'file_name':item['file_name'], 'page_number':item['page_number'], 'line_number': item['line_number'], 'language': item['language']} for item in dataset]
        else:
            self.train.text_lang = None
        
        if filter_last_doc is None:
            self.number_years = 1
        else:
            self.number_years = abs(filter_last_doc)
        
        print('Number of years',self.number_years)

        print('Columns', [item for item in self.dataset[0]] )
        print('Finishing data loading', Counter([item.get('risk_desc') for item in self.dataset if item.get('risk_desc') is not None]))
        print('Number companies', len({item.get('company_id') for item in self.dataset}))
        print('Number documents', len({item.get('file_name') for item in self.dataset}))

        return True

    def removeEMails(self, text, tagReplace=''):
        emailRegex = r"([a-z]*[_-]?[a-z]+@[a-z]+.?[a-z]{0,3}.?[a-z]{2})"
        text = re.sub(emailRegex, tagReplace, text)
        return text

    def removeNoisyCharacters(self, text, tagReplace=''):
        same_conseqRegex = r"(?:[^\s,.;°]*([^\s0-9w])\1{2,}[^\s,.;]*)" #more than three consecutives characters different than numbers and spaces
        text = re.sub(same_conseqRegex, tagReplace, text) 
        same_conseqRegex2 = r"([-_/.«»]\s?){3,}" #consecutive characters with blank space in the middle
        text = re.sub(same_conseqRegex2, tagReplace, text)
        noisyRegex = r"[\“\”\‘'|—_\(\)«»:;-]"
        text = re.sub(noisyRegex, tagReplace, text)
        return text 

    def removeRedundantTags(self, text, tagSearch): 
        tag_regex = r"(\[+" + tagSearch[1:-1] + r"\]+\s?){2,}"
        text = re.sub(tag_regex, tagSearch, text) 
        return text

    def removeApostropheDash(self, text, tagReplace=' '):
        noisyRegex = r"['’`-]"
        text = re.sub(noisyRegex, tagReplace, text)
        return text

    def splitWithCharacters(self, text, tagReplace='\n', keep_numbers=False):
        if keep_numbers:
            noisyRegex = r"([;:}{!?#$\(\)\{\}\"&\*\+@\^|=~-])+" #Remove special characters except dot and colon
            text = re.sub(noisyRegex, tagReplace, text)      
            noisyRegex = r"[\.,]+(?=[^0-9]|$)"       #remove dot and colon if not followed by number
            text = re.sub(noisyRegex, tagReplace, text)      
        else:
            noisyRegex = r"([\.,]+[\s](?!\d)(?!,)|([;:}{!?#$\(\)\{\}\"&\*\+@\^|=~-]))+" 
            text = re.sub(noisyRegex, tagReplace, text) 
        return text

    def removeMultipleBlankSpaces(self, text, tagReplace=' '):
        noisyRegex = r"(\s){2,}"
        text = re.sub(noisyRegex, tagReplace, text) 
        return text

    def removeEnumerators(self, text, tagReplace =''):
        enumeratorRegex = r"(?:\s{0,3})(?:i?[xv]?[a-kxv]{1,3})[.)]\s"
        text = re.sub(enumeratorRegex, tagReplace, text)
        return text

    def removeNumberInText(self, text, tag_replace=None, normalization_value = None):
        if tag_replace is None: tagReplace = self.special_tokens["number"]
        if normalization_value is None or len(normalization_value) ==0 or normalization_value.strip()=="0": normalization_value = 1
        newText = " " + text + " "
        nRegex1_percentage = r" (\d?[.,]?\d*\s?\%\s)"  #number that ends with %
        nRegex2_telephone = r"(\(\+\d{2,4}\)[\s\d]{6,14})\d\s|\+[\s\d]{6,14}\d" # numbers with international code
        nRegex3_codification = r"\b[a-zA-Z]{1,3}\d{2,100}|\d{2,100}[a-zA-Z]{1,3}|[a-zA-Z]{1,3}\d{2,100}[a-zA-Z]{1,3}\b" #numbers which start and/or ends with chars (3)
        nRegex4_rawnumber = r"(-?\s?\b\d{0,3}[.,]?\d{1,3}[.,]\d{1,2}(?!\%))\b" #numbers with decimals
        nRegex5_rawnumber = r"(-?\s?\d{0,3}[.,'\s]?\d{1,3}[.,\s]\d{3}(?!\%)\s)" #numbers more than 1K with no decimals
        #old: "([,.;]?\(?\+?[0-9]{0,1}\)?\s?[0-9]?[0.,']?[0-9]+[.,]?[0-9]{0,3}[,.;]?)" 
        newText = self.replaceCoincidences(newText, nRegex1_percentage, self.special_tokens['percentage'])
        newText = self.replaceCoincidences(newText, nRegex2_telephone, self.special_tokens['phone'])
        newText = self.replaceCoincidences(newText, nRegex3_codification, self.special_tokens['code'])
        #custom numerical replacement
    
        coincidences = re.findall(nRegex4_rawnumber, newText.lower())
        coincidences2 = re.findall(nRegex5_rawnumber, newText.lower())
        coincidences.extend(coincidences2)
        if len(coincidences) > 0:
            coincidences = {c:0 for c in coincidences if len(c) >0 }
            for coincidence in coincidences:
                try:
                    decimal = coincidence.strip()[-3:] + ".,"
                    index_removal = min(decimal.index("."),decimal.index(","))
                    decimal = decimal[:index_removal]
                    restnumber = coincidence.strip()[:-3]
                    restnumber = restnumber.replace(".","").replace(",","").replace("'","") + decimal
                    number_base = float(restnumber.strip())
                    if normalization_value is not None:
                        norm_value = math.log10(abs(number_base/float(normalization_value.strip())))
                        added_token = ""
                        if number_base >0:
                            added_token = self.special_tokens['positive']
                        else:
                            added_token = self.special_tokens['negative']

                        if norm_value > 0:
                            tagReplace = added_token + ' ' + self.special_tokens["big_number"+ str(round(norm_value))]
                        elif norm_value < 0:
                            tagReplace = added_token + ' ' + self.special_tokens["small_number"+ str(abs(round(norm_value)))]                            
                except:
                    pass

                newText = newText.replace(coincidence.strip(), ' ' + tagReplace + ' ').strip()
        
        return newText

    def replaceCoincidences(self, text, regex, tagReplace, min_size=0):
        newText = text
        coincidences = re.findall(regex, text)
        if len(coincidences) > 0:
            if type(coincidences[0])==str:
                coincidences = {c.strip():0 for c in coincidences if len(c)  >min_size }
            else:   
                coincidences = {t.strip():0 for c in coincidences for t in c if len(t) > min_size}

            for coincidence in coincidences: 
                newText = newText.replace(coincidence.strip(), tagReplace).strip()
        
        return newText

    def removeDatesInText(self, text, tagReplace = ''):
        if tagReplace is None: tagReplace = self.special_tokens["date"]

        patternShortDates = "(\d{1,2}\s?[-./]\s?\d{1,2}\s?[-./]\s?\d{2,4})"
        patternYear4Digits = "(\s[,.;]?[0-3]?[0-9][-./|\s]\s*(?:[a-zÀ-Ÿ']*([0-3]?[1-9])?){1}[-./|\s]\s*[12][089][0-9]{2}[,.;]?\s)"
        patternYear2Digits = "(\s[,.;]?[0-3]?[0-9][-./|\s]\s*(?:[a-zÀ-Ÿ']*([0-3]?[1-9])?){1}[-./|\s]\s*[0-9]{1,2}[,.;]?\s)"
        newText =  ' ' + text + ' '
        self.replaceCoincidences(newText, patternShortDates, tagReplace, min_size=5)
        newText =self.replaceCoincidences(newText, patternYear4Digits, tagReplace, min_size=5)
        newText =self.replaceCoincidences(newText, patternYear2Digits, tagReplace, min_size=5)

        return newText

    def cleanRecord(self, info_text, lower_case = True):
        original_text = info_text['text_page']
        remove_numbers = info_text['remove_numbers']
        normalize = info_text['normalize']
        normalization_value = info_text['normalization_value']
        info_text['index'] = info_text['index'] if info_text.get('index') is not None else 0
        if lower_case:
            original_text = original_text.lower()

        cleaned_text_lines = ''
        text_page = self.removeEMails(original_text, tagReplace= "<tempo>")
        text_page = self.removeMultipleBlankSpaces(text_page) 

        text_lines = text_page.split('\n') 
        for i_line, n_text_line in enumerate(text_lines):
            if len(n_text_line.split(' ')) > 3 and len(n_text_line)>5:
                n_text_line = self.removeDatesInText(n_text_line, tagReplace= None)
                n_text_line = self.removeApostropheDash(n_text_line, tagReplace= " ")
                n_text_line = self.removeEnumerators(n_text_line, tagReplace= " ")
                n_text_line = self.removeNoisyCharacters(n_text_line, tagReplace= " ")
                if remove_numbers or normalize:
                    n_text_line = self.removeNumberInText(n_text_line, normalization_value = normalization_value) 
                n_text_line = self.splitWithCharacters(n_text_line, tagReplace= "<tempo>", keep_numbers= not remove_numbers).strip()

                n_text_line_list = n_text_line.split('<tempo>')
                if len(n_text_line_list)>1:
                    n_text_line_list = [item.strip() for item in n_text_line_list if len(item.strip())>0] 
                    cleaned_text_lines += ' '.join(n_text_line_list) + "\n"
                else:
                    cleaned_text_lines += n_text_line  + "\n"
        
        if len(cleaned_text_lines) == 0:
            cleaned_text_lines = original_text + "\n"

        return {'text':cleaned_text_lines[:-1],'index':info_text['index']}
    
    def preprocessData(self, text_list, distrib_tool, normalization_list = None, remove_numbers=True): 
        if normalization_list is None: 
            normalize = False
        else:
            normalize = True

        if remove_numbers and not normalize: 
            print('Removing numbers while cleaning')
        else:
            if not normalize:
                print('Numbers are not removed while cleaning')
            else:
                print('Numbers are going to be normalized')
        
        number_workers = 1#mp.cpu_count()
        cleaned_list = [] #['' for item in range(len(text_list))]

        for i_page in tqdm.tqdm(range(0,len(text_list), number_workers)):  #parallelization of cleaning
            normalization_value = 1 if normalization_list is None else normalization_list[i_page]
            info_text = {'text_page': text_list[i_page].lower(), 'remove_numbers':remove_numbers, 'normalize':normalize,'normalization_value':normalization_value, 'index':i_page}
            
            '''
            dataset_batch = []
            for i, text_page in enumerate(text_list[i_page:i_page+number_workers]):
                info_text = {'text_page': text_page.lower(), 'remove_numbers':remove_numbers, 'normalize':normalize,'normalization_value':normalization_value, 'index':i_page+i}
                dataset_batch.append(info_text) 
            
            results = distribute(distrib_tool, dataset_batch, self.cleanRecord, total_workers=number_workers)
            for result in results:
                cleaned_list[result['index']]= result['text']
            dataset_batch = []
            '''

            cleaned_list.append(self.cleanRecord(info_text)['text'])

        return cleaned_list
    
    def prepare_input_data(self, model_dir, model_type, group_level, y_label_type, labels, y_top_n, trim_begining, tokenizer, sequence_strategy, sequence_shift, max_segments_bert, max_proportion_numbers_words=0.2, filter_min_words=20, terms_filter_path=None, clean_data = None, remove_numbers = True,time_grouped_by_company=False, normalization_column=None, worker_workload=None, distibution_tool=None):
        '''
        group_level: page, document, paragraph, line
        y_label_type: Column name of dataset: i.e. document_type, company_name, status. If "group" is specified, then is the group name, ex. subtitle for paragraphs 
        sequence_strategy: 
        max_segments_bert: Max Number of segments, if 1, then is only a BERT model, if > 0 is a BERT + LSTM model 
        '''
        
        if model_type == "SA":
            group_level = 'document'
            if y_label_type is None: y_label_type='risk_desc'
            y_top_n = None 
            max_proportion_numbers_words =0.5
        elif model_type == "NER":
            group_level = 'paragraph'
            y_label_type='group'
            max_segments_bert = 1
            if y_top_n is None:
                y_top_n = 500 
            
        if y_label_type is not None:
            print('Filter label', y_label_type)
            print('Group Level', group_level)
            if labels is not None:
                labels = [item.lower() for item in labels]
                for i_label, label in enumerate(labels):
                    if label[0]=="_":
                        labels[i_label] = ' '.join(label.split("_")).strip()
                print('Labels:', labels)

        max_sequence_length = 512 #BERT model Limitation
       
        additional_headers = []
        normalization_list = None 

        if normalization_column is not None:
            additional_headers.append(normalization_column)
        
        parallel_workers = mp.cpu_count() if distibution_tool is not None else 1

        self.get_grouped_text(rows_list=self.dataset, clean_data=clean_data, remove_numbers=remove_numbers, group_level=group_level, y_label_type=y_label_type, filter_min_words=filter_min_words, additional_headers = additional_headers,parallel_workers=parallel_workers, distrib_tool=distibution_tool, workload=worker_workload)
        print('label',y_label_type,'min_words',filter_min_words,'group_level',group_level)
        print('Labels', Counter([item.get('risk_desc') for item in self.dataset if item.get('risk_desc') is not None]))
        print('Number of companies', len({item['company_id'] for item in self.dataset}))

        if clean_data or normalization_column is not None:
            if clean_data is None: print('Cleaning data')
            if normalization_column is not None: 
                print('Normalizing numbers with column', normalization_column)
                normalization_list = [item.get(normalization_column) for item in self.train.all_metadata]
            self.train.all_words = self.preprocessData(self.train.all_words, distibution_tool, normalization_list = normalization_list, remove_numbers=remove_numbers)
        else:
            print('Data is not going to be cleaned')

        terms_filter_list = None
        if terms_filter_path is not None:
            if os.path.exists(terms_filter_path):
                with open(terms_filter_path,'r') as f:
                    terms_filter_list = json.load(f)
            else:
                print('Filter file not found on',terms_filter_path)


        if tokenizer is None: 
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased") 
        elif type(tokenizer) == str:
            if tokenizer.lower() == 'finbert':
                tokenizer =  AutoTokenizer.from_pretrained("ProsusAI/finbert")
                print('Loading tokenizer with FinBERT')
            else:
                tokenizer = BertTokenizer.from_pretrained(tokenizer)
                print('Loading tokenizer from pretained', tokenizer)
        
        if labels is not None:
            labels_dict = {item:0 for item in self.train.all_labels if item is not None and len(item)>0 and item in labels}
        else:
            labels_dict = {item:0 for item in self.train.all_labels if item is not None and len(item)>0}
        labels_dict = {item:i for i,item in  enumerate(labels_dict)}
        is_reduced = False
        
        #reduce the data to the TOP N most frequent
        if y_top_n is not None and y_top_n < len(labels_dict):
            print('reducing classes from ',len(labels_dict), 'to', y_top_n)                 
            temp_classes = [re.sub(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿ-'/]+", '', item) for item in self.train.all_labels]
            temp_classes = [item for item in temp_classes if len(item)>=5 ]
            c_label_classes = Counter(temp_classes).most_common(y_top_n)
            labels_dict = {item[0]:i for i,item in enumerate(c_label_classes)}
            if self.loaded_labels is not None:
                print('Number of loaded labels:', type(self.loaded_labels))
                print(len(self.loaded_labels))
                total_current_labels = len(labels_dict)
                if len(self.loaded_labels) != len(self.train.all_labels) and len(self.loaded_labels)>0: 
                    for i_label, label in enumerate(self.train.all_labels): 
                        if self.loaded_labels.get(label) not in ['Others','Finance','Deposits']:
                            self.train.all_labels[i_label] = self.loaded_labels.get(label)
                        else:
                            self.train.all_labels[i_label] = None
                        
                    c_label_classes = Counter(self.train.all_labels).most_common(y_top_n)
                    labels_dict = {item[0]:0 for item in c_label_classes if item[0] is not None}
                    labels_dict = {item:i for i,item in enumerate(labels_dict)}
                    total_new_labels = len(labels_dict)
                    print('Replaced ', total_current_labels, " classes with" , total_new_labels ,"classes")        
            
            is_reduced = True
        else:   
            for i_label, label in enumerate(self.train.all_labels):  
                if labels is not None and labels_dict.get(label) is None:
                    self.train.all_labels[i_label] = None 

        if not self.is_train:
            saved_labels= self.get_training_classes(model_dir)
            self.train.labels2idx, self.train.idx2labels, is_reduced = saved_labels['labels2idx'],saved_labels['idx2labels'],saved_labels['is_reduced']
            self.train.idx2labels = {int(item):self.train.idx2labels[item] for item in self.train.idx2labels}
        else:     
            self.train.labels2idx = labels_dict
            self.train.idx2labels= {i:item for i,item in  enumerate(labels_dict)}
            self.save_training_classes(model_dir, {'labels2idx':self.train.labels2idx,'idx2labels':self.train.idx2labels,'is_reduced':is_reduced})
        
        padtypes = {'words':tokenizer.pad_token, 'token_type_ids':1, 'attention_mask':0, 'input_ids':tokenizer.pad_token_id}
        #edgetokens = {'cls_ii':None, 'sep_ii':None,'cls_tt':None, 'sep_tt':None,'cls_am':None, 'sep_am':None}

        #featured_data={'encoded':[],'text':[],'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'words': [], 'labels':[]}
        i_row = -1 
        has_removed_records = False
        print('Preparing input data')               

        total_filtered_out = 0
        companies_to_remove = {}
        for sample in self.train.all_words:
            i_row += 1
            
            if terms_filter_list is not None:
                found_term = False
                for term_base in terms_filter_list:
                     if term_base in self.train.all_words[i_row]:
                        found_term = True
                        break
                
                if not found_term:
                    self.train.all_labels[i_row] = None
                    total_filtered_out +=1

            #remove text which have a significant amount of numbers
            if max_proportion_numbers_words > 0 and max_proportion_numbers_words < 1:
                proportion_numbers_words = getProportionNumbers(self.train.all_words[i_row])
                if proportion_numbers_words > max_proportion_numbers_words:
                    self.train.all_labels[i_row] = None
                    companies_to_remove[self.train.all_metadata[i_row]['company_id']] = True
                    total_filtered_out +=1

            if not time_grouped_by_company:
                if self.train.all_labels[i_row] is None  or (is_reduced and  self.train.labels2idx.get(self.train.all_labels[i_row]) is None) or (not is_reduced and len(str(self.train.all_labels[i_row]))==0):
                    companies_to_remove[self.train.all_metadata[i_row]['company_id']] = True
                    self.train.all_words[i_row] = None
                    self.train.all_metadata[i_row] = None 
                    self.train.text[i_row] = None
                    self.train.text_lang[i_row] = None
                    has_removed_records = True
                    continue            
            else:
                last_doc_company = self.dataset_last_documents.get(self.train.all_metadata[i_row]['company_id']) 
                if last_doc_company is None:
                    companies_to_remove[self.train.all_metadata[i_row]['company_id']] = True
                    self.train.all_words[i_row] = None
                    self.train.all_metadata[i_row] = None 
                    self.train.text[i_row] = None
                    self.train.text_lang[i_row] = None
                    self.train.all_labels[i_row] = None
                    has_removed_records = True
                    continue   
            
        if total_filtered_out > 0:
            print('There are ', total_filtered_out, 'filtered out records because does not contains any financial term')

        if has_removed_records:
            print('Data was removed')
            if time_grouped_by_company:
                for i_label, record_label in enumerate(self.train.all_labels): #removing multiyear data which company was affected
                    if record_label is None or companies_to_remove.get(self.train.all_metadata[i_label]['company_id']) is not None:
                        self.train.all_words[i_label] = None 
                        self.train.all_metadata[i_label] = None 
                        self.train.text[i_label] = None 
                        self.train.text_lang[i_label] = None 
                docs_per_cia = {item['company_id']:0 for item in self.train.all_metadata}
                for company_id in docs_per_cia:
                    docs_per_cia[company_id] = len([1 for item in self.train.all_metadata if item['company_id']==company_id])

                assert len(docs_per_cia) == len([item for item in docs_per_cia if docs_per_cia[item]==self.number_years]) 

            self.train.all_labels = [item for item in self.train.all_labels if item is not None] 
            self.train.all_words = [item for item in self.train.all_words if item is not None] 
            self.train.all_metadata = [item for item in self.train.all_metadata if item is not None] 
            self.train.text = [item for item in self.train.text if item is not None] 
            self.train.text_lang = [item for item in self.train.text_lang if item is not None] 
        else:
            print('No data was removed')

        if len(self.train.all_labels) != len(self.train.all_metadata):
            print('All labels size:', len(self.train.all_labels))
            print('All metadata size:', len(self.train.all_metadata))

        temp_featured_data = {}
        if parallel_workers ==1: worker_workload=1
        for i_sample in tqdm.tqdm(range(0,len(self.train.all_words),parallel_workers*worker_workload)):
            if parallel_workers ==1:
                sample = self.train.all_words[i_sample]
                sample_label = self.train.all_labels[i_sample]
                results = tokenize_text(tokenizer, sample, sample_label, sequence_strategy, max_sequence_length,trim_begining, self.train.labels2idx)
                if len(temp_featured_data) ==0:
                    temp_featured_data = results
                    temp_featured_data['all_metadata'] = [self.train.all_metadata[i_sample]] 
                    temp_featured_data['all_labels'] = [sample_label]
                    temp_featured_data['text_lang'] = [self.train.text_lang[i_sample]]
                else:
                    for feature_type in results:
                        temp_featured_data[feature_type].extend(results[feature_type])
                    temp_featured_data['all_metadata'].append(self.train.all_metadata[i_sample]) 
                    temp_featured_data['all_labels'].append(sample_label)
                    temp_featured_data['text_lang'].append(self.train.text_lang[i_sample])
            else:
                batch_info = []
                for i_worker in  range(0,parallel_workers):
                    i_start = i_sample+i_worker*worker_workload
                    i_end = i_sample+(i_worker+1)*worker_workload
                    all_data = {
                        'all_words':self.train.all_words[i_start:i_end],
                        'all_labels':self.train.all_labels[i_start:i_end],
                        'all_metadata':self.train.all_metadata[i_start:i_end],
                        'text':self.train.text[i_start:i_end],
                        'text_lang':self.train.text_lang[i_start:i_end],
                    } 
                    info = {'all_data':all_data,'tokenizer':tokenizer,'sequence_strategy':sequence_strategy,'max_sequence_length':max_sequence_length,'trim_begining':trim_begining,'labels2idx':self.train.labels2idx} 
                    batch_info.append(info)

                results = distribute(distibution_tool, batch_info, remote_tokenize_text, total_workers=parallel_workers)
                for result in results:
                    if len(temp_featured_data)==0:
                        temp_featured_data=result
                    else:
                        for feature_type in result:
                            temp_featured_data[feature_type].extend(result[feature_type])
        
        self.train.all_tokens, self.train.all_input_ids,self.train.all_segment_ids,self.train.all_input_mask,self.train.all_label_ids  = [],[],[],[],[]

        self.train.all_labels,self.train.all_words,self.train.all_metadata,self.train.text,self.train.text_lang = [],[],[],[],[] 


        print('Finished preparing data.')
        if sequence_strategy == 'document_batch':
            print('Starting preparing batchs by document')
            #left_shift  = max_sequence_length - sequence_shift
            
            segments_list = []
            overloaded_segments = {} 
            tokens_per_word_list= []
            tokens_per_group = []
            splited_words_per_group = []
            words_per_group = []
            fertility = []
            fertility_per_group = [] 
            number_segments_total = 0
            #determine the number of segments of each document based on  max_sequence_length
            for i_sample, sample_metadata  in  enumerate(temp_featured_data['all_metadata']):
                word_group = temp_featured_data['words'][i_sample] 

                if time_grouped_by_company and self.dataset_last_documents.get(sample_metadata['company_id']) is None:
                    continue
                proportion_numbers_words = getProportionNumbers(temp_featured_data['text'][i_sample])
                number_segments = 1 if len(word_group) < max_sequence_length else int(np.ceil(len(word_group) / sequence_shift))
                number_segments_total += number_segments
                if number_segments > 1 and (number_segments-1) *max_sequence_length>=len(word_group):
                    number_segments -= 1    
                    number_segments_total -= 1 
                if number_segments > max_segments_bert:
                    if overloaded_segments.get(number_segments) is None: overloaded_segments[number_segments]=0 
                    overloaded_segments[number_segments] +=1
                    number_segments = max_segments_bert 
                segments_list.append(number_segments)        
                #statistics
                number_words = len(temp_featured_data['text'][i_sample].split(' '))
                number_tokens = len(word_group)

                if number_words >0:
                    words_per_group.append(number_words)
                    tokens_per_word_list.append(number_tokens/number_words)
                    tokens_per_group.append(number_tokens)
                    consecutive_subwords = 0
                    total_splited_words = 0
                    for token in word_group:
                        if token[:2] == "##":
                            consecutive_subwords += 1
                        else:
                            if consecutive_subwords == 1:
                                total_splited_words += 1
                            
                            if consecutive_subwords >0:
                                fertility.append(consecutive_subwords)
                            consecutive_subwords = 0
                    
                    if consecutive_subwords >0:
                        fertility.append(consecutive_subwords)
                    fertility_per_group.append((number_words+sum(fertility))/number_words)
                    if total_splited_words >0: splited_words_per_group.append(total_splited_words/number_words)

            if len(overloaded_segments) >0: 
                trimmed_segments_total = sum([item*overloaded_segments[item] for item in overloaded_segments])
                ratio_trimmed = trimmed_segments_total / number_segments_total
                print('Total segments:', number_segments_total, 'Trimed:', trimmed_segments_total,'ratio',ratio_trimmed)
            else:
                print("All the data fit into the max_number_segments",max_segments_bert)

            if len(tokens_per_word_list) > 0: print('Average Tokens per word',sum(tokens_per_word_list)/len(tokens_per_word_list)) 
            if len(tokens_per_group) > 0: print('Average Tokens per group',sum(tokens_per_group)/len(tokens_per_group)) 
            if len(splited_words_per_group) > 0: print('Average divided words per group',sum(splited_words_per_group)/len(splited_words_per_group)) 
            if len(words_per_group) > 0: print('Average words per group',sum(words_per_group)/len(words_per_group)) 
            if len(fertility_per_group) >0: print('Average fertility per group', sum(fertility_per_group)/len(fertility_per_group))
            
            featured_data_words_temp = []
            featured_data_labels_temp = []
            featured_data_input_ids_temp = []
            featured_data_token_types_temp = []
            featured_data_attention_masks_temp = []
            start_indexes = {}
            clean = 0
            companies_for_removing = {}
            
            print('Number of labels', len( temp_featured_data['all_labels']), "metadata:", len(temp_featured_data['all_metadata']))
            print('Segments to remove (short ones or too many numbers)', len([item for item in segments_list if item==0]))
            print('Starting generation of input ids')
            for i_doc in tqdm.tqdm(range(len(temp_featured_data['words']))):  
                if segments_list[i_doc] ==0 or temp_featured_data['all_labels'][i_doc] is None: #removing short segments or segments with too many numbers
                    companies_for_removing[temp_featured_data['all_metadata'][i_doc]['company_id']] = temp_featured_data['all_labels'][i_doc]
                    temp_featured_data['text'][i_doc] = None 
                    temp_featured_data['text_lang'][i_doc] = None
                    temp_featured_data['words'][i_doc]  = None 
                    temp_featured_data['all_labels'][i_doc]  = None
                    temp_featured_data['all_metadata'][i_doc]  = None
                    featured_data_labels_temp.append(None)
                    has_removed_records = True  
                    clean +=1
                else:
                    start_indexes[temp_featured_data['all_metadata'][i_doc]['document']]= len(featured_data_words_temp)
                    word_group = temp_featured_data['words'][i_doc]
                    #divide document's text into the pre-calculated number of segments
                    for i_segment in range(segments_list[i_doc]):
                        start_i = i_segment * sequence_shift
                        end_i = i_segment * sequence_shift + max_sequence_length - 2
                        featured_data_words_temp.append([tokenizer.cls_token] + word_group[start_i:end_i] + [tokenizer.sep_token]) 
                        featured_data_input_ids_temp.append([tokenizer.cls_token_id] + temp_featured_data['encoded'][i_doc]['input_ids'][1:-1][start_i:end_i] + [tokenizer.sep_token_id] )
                        featured_data_token_types_temp.append([temp_featured_data['encoded'][i_doc]['token_type_ids'][0]] + temp_featured_data['encoded'][i_doc]['token_type_ids'][1:-1][start_i:end_i]+ [temp_featured_data['encoded'][i_doc]['token_type_ids'][0]] )
                        featured_data_attention_masks_temp.append([temp_featured_data['encoded'][i_doc]['attention_mask'][0]] + temp_featured_data['encoded'][i_doc]['attention_mask'][1:-1][start_i:end_i]+ [temp_featured_data['encoded'][i_doc]['attention_mask'][0]] )
                        assert len(featured_data_words_temp[-1])==len(featured_data_input_ids_temp[-1])
                        assert len(featured_data_words_temp[-1])==len(featured_data_token_types_temp[-1])
                        assert len(featured_data_words_temp[-1])==len(featured_data_attention_masks_temp[-1])
                    ## pad right                
                    total_to_pad = max_sequence_length-len(featured_data_words_temp[-1])
                    if total_to_pad>0:
                        featured_data_words_temp[-1].extend([padtypes['words'] for item in range(total_to_pad)])             
                        featured_data_input_ids_temp[-1].extend([padtypes['input_ids'] for item in range(total_to_pad)])  
                        featured_data_token_types_temp[-1].extend([padtypes['token_type_ids'] for item in range(total_to_pad)])  
                        featured_data_attention_masks_temp[-1].extend([padtypes['attention_mask'] for item in range(total_to_pad)])  
                    
                    ## pad bottom
                    pad_array = [padtypes['words'] for item in range(max_sequence_length)]
                    for i in range(max_segments_bert - segments_list[i_doc]):
                        featured_data_words_temp.append(pad_array)
                        featured_data_input_ids_temp.append( [padtypes['input_ids'] for item in range(max_sequence_length)])
                        featured_data_token_types_temp.append( [padtypes['token_type_ids'] for item in range(max_sequence_length)])
                        featured_data_attention_masks_temp.append( [padtypes['attention_mask'] for item in range(max_sequence_length)])
                        
                    #only one label per sequence
                    featured_data_labels_temp.append(temp_featured_data['labels_ids'][i_doc])
                        
            for i_t, test1 in enumerate(featured_data_words_temp):
                assert len(test1) == max_sequence_length 

            if clean>0:
                featured_data_labels_temp = [item for item in featured_data_labels_temp if item is not None]
                temp_featured_data['all_labels'] = [item for item in temp_featured_data['all_labels'] if item is not None]

                temp_featured_data['words'] = [item for item in temp_featured_data['words'] if item is not None] 
                temp_featured_data['all_metadata'] = [item for item in temp_featured_data['all_metadata'] if item is not None] 
                temp_featured_data['text'] = [item for item in temp_featured_data['text'] if item is not None] 
                temp_featured_data['text_lang'] = [item for item in temp_featured_data['text_lang'] if item is not None] 
                
            if time_grouped_by_company:
                documents_to_remove = 0
                print('Removing years that are not in the scope')
                print('Companies to remove', len(companies_for_removing))
                for i_metadata, item_metadata in enumerate(temp_featured_data['all_metadata']):
                    if item_metadata is not None and companies_for_removing.get(item_metadata['company_id']) is not None:
                        documents_to_remove +=1
                        temp_featured_data['all_labels'][i_metadata] = None 
                        temp_featured_data['all_metadata'][i_metadata]  = None 
                        temp_featured_data['text'][i_metadata]  = None
                        temp_featured_data['text_lang'][i_metadata] = None
                        temp_featured_data['words'][i_metadata]  = None 
                        featured_data_labels_temp[i_metadata] = None
                        
                        start_index = start_indexes[item_metadata['document']]
                        end_index = start_index + max_segments_bert
                        for i_index in range(start_index, end_index):
                            featured_data_words_temp[i_index] = None
                            featured_data_input_ids_temp[i_index] = None
                            featured_data_token_types_temp[i_index] = None
                            featured_data_attention_masks_temp[i_index] = None 
                print('Documents to remove', documents_to_remove)

            print('total to remove:', len([item for item in featured_data_words_temp if item is None]))
            print('New total:', len([item for item in featured_data_words_temp if item is not None]))
            self.train.all_tokens = [item for item in featured_data_words_temp if item is not None] 
            self.train.all_input_ids  = [item for item in featured_data_input_ids_temp if item is not None]
            self.train.all_segment_ids = [item for item in featured_data_token_types_temp if item is not None]
            self.train.all_input_mask  = [item for item in featured_data_attention_masks_temp if item is not None]

            self.train.all_label_ids = [item for item in featured_data_labels_temp if item is not None]
            self.train.all_labels = [item for item in temp_featured_data['all_labels'] if item is not None]

            self.train.all_words = [item for item in temp_featured_data['words'] if item is not None] 
            self.train.all_metadata = [item for item in temp_featured_data['all_metadata'] if item is not None] 
            self.train.text = [item for item in temp_featured_data['text'] if item is not None] 
            self.train.text_lang = [item for item in temp_featured_data['text_lang'] if item is not None]  

        self.args_ret['lstm_sequence_length'] = max_segments_bert
        if time_grouped_by_company:
            self.args_ret['lstm_stacked_layers'] = self.number_years
        else:
            self.args_ret['lstm_stacked_layers'] = 1

        self.args_ret['sequence_shift'] = sequence_shift
        
        if max_segments_bert >1:
            self.args_ret['is_bert_lstm'] = True
            print('Using BERT + LSTM')
        else:
            self.args_ret['is_bert_lstm'] = False
            print('Using BERT')

        
        #calculate weights
        if len(self.train.all_labels)==0:
            raise Exception('There is no labeled data.')
        
        label_set=[item for item in self.train.all_labels if item is not None]
        #print(label_set)
        print('Counter after preparing data:', Counter(self.train.all_labels))
        class_weights = compute_class_weight(class_weight ='balanced', classes =np.unique(label_set),  y =label_set)
        self.train.class_weights_dict = dict(zip(np.unique(label_set), class_weights))
        self.train.weights = torch.tensor(class_weights,dtype=torch.float)    
        
        self.args_ret['tokenizer'] = tokenizer

        return self.args_ret

    def get_grouped_text(self, rows_list, clean_data, remove_numbers, group_level, y_label_type, filter_min_words, additional_headers=[], number_records=None, parallel_workers=1, workload=1,distrib_tool=None):
        if len(rows_list)==0:
            raise Exception('There is no data to group')

        if y_label_type is None: 
            if group_level == 'page': y_label_type = 'doc_page'
            elif group_level == 'document': y_label_type = 'file_name' 
            else: y_label_type = 'doc_page'  

        #last_document = rows_list[0]['file_name']
        #last_page = rows_list[0]['page_number']
        #accumulated_text = ''
        self.train.all_words, self.train.all_labels, self.train.all_metadata, self.train.text, self.train.text_lang = [],[],[],[],[]
        #temp_text_lang = []
        #temp_text = []
        
        #last_enumerator = ''
        print('Grouping text by ', group_level)
        if parallel_workers is None or parallel_workers <=1: parallel_workers = 1
        if workload is None or workload <=1 or parallel_workers==1: workload=1
        document_list = {item['file_name']:0 for item in rows_list}
        document_list = [item for item in document_list]
        for i_document in tqdm.tqdm(range(0,len(document_list), parallel_workers*workload)):
            if parallel_workers ==1:
                document_name = document_list[i_document]
                row_list_document = [row for row in rows_list if row['file_name']==document_name]  
                last_cia_document = self.dataset_last_documents.get(row_list_document[-1]['company_id'])
                temp_metadata, temp_all_words, temp_text_lang, temp_text, temp_all_labels = process_group_by_document(group_level, row_list_document, document_name, filter_min_words, additional_headers, y_label_type, last_cia_document)
                self.train.all_words.extend(temp_all_words)
                self.train.all_labels.extend(temp_all_labels)
                self.train.all_metadata.extend(temp_metadata)
                self.train.text_lang.extend(temp_text_lang)
                self.train.text.extend(temp_text)
            else:
                batch_info = []
                for j_document in  range(0,parallel_workers*workload,workload):
                    worker_list = document_list[i_document+j_document:i_document+j_document+workload]
                    row_list_documents = [row for row in rows_list if row['file_name'] in worker_list]  
                    company_names = {item['file_name']:item['company_id'] for item in row_list_documents}
                    last_cia_documents = {item:self.dataset_last_documents.get(company_names.get(item)) for item in worker_list}
                    info = {'group_level':group_level, 'row_list_document':row_list_documents, 'document_list': worker_list, 'filter_min_words':filter_min_words, 'additional_headers': additional_headers, 'y_label_type':y_label_type, 'last_cia_documents':last_cia_documents}
                    batch_info.append(info)

                results = distribute(distrib_tool, batch_info, remote_process_group_by_document, total_workers=parallel_workers)
                
                for result in results:
                    if len(result)>0:
                        self.train.all_metadata.extend(result[0])
                        self.train.all_words.extend(result[1])
                        self.train.text_lang.extend(result[2])
                        self.train.text.extend(result[3]) 
                        self.train.all_labels.extend(result[4]) 
                        

            if number_records is not None and len(self.train.text) >= number_records:
                break

        #if len(self.train.text_lang) > 0:
        #    self.train.text_lang =  temp_text_lang
        #    self.train.text =  temp_text   

        if clean_data or remove_numbers:
            for i_record, record_text in enumerate(self.train.text):
                self.train.text[i_record] = self.cleanRecord({'text_page':record_text,'remove_numbers':remove_numbers,'normalize':False,'normalization_value':None})

        print('Finishing grouping with', len([item for item in self.train.all_labels if item is not None and len(item)>0]), group_level)

    def save_training_classes(self, model_dir, training_classes):    
        with open(os.path.join(model_dir,'training_classes.json'),'w') as f:
            json.dump(training_classes, f)

    def get_training_classes(self, model_dir):
        with open(os.path.join(model_dir,'training_classes.json'),'r') as f:
            training_classes = json.load(f)
        return training_classes 
    
    def get_dataset_name(self):    
        if self.dataset_name is not None:
            return self.dataset_name   
        else:
            if self.is_train:
                suffix = 'train'
            else:
                suffix = 'test'
            
            if self.total_records is None: 
                number = 'all'
            else:
                if np.abs(self.total_records) >= 1000:
                    number = str(int(np.abs(self.total_records) /1000)) + "K"
                else:
                    number = str(np.abs(self.total_records))
            return os.path.join(self.output_dir, 'dataset_' + number + "_" + suffix + ".pickle")            

    def save_dataset(self, dataset_name = None, save_full_dataset=False):
        if dataset_name is None: 
            file_name = self.dataset_name 
        else:
            file_name = dataset_name

        if save_full_dataset:
            dataset = {'train':None, 'val':None, 'is_train':True, 'args_ret':None, 'dataset':self.dataset} 
        else:
            if self.is_train:
                dataset = {'train':self.train, 'val':self.val, 'is_train':True, 'args_ret':self.args_ret}
            else:
                dataset = {'test':self.test, 'is_train':False, 'args_ret':self.args_ret}

        with open(file_name, 'wb') as f:
            pickle.dump(dataset, f)      
            print('Dataset saved in', file_name)
            
    def load_dataset_from_file(self, dataset_name=None): 
        file_name =  dataset_name if dataset_name is not None else self.dataset_name
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f: 
                dataset = pickle.load(f)
                print('Loading dataset from ',file_name)
                
            if dataset['is_train']:
                self.train = dataset['train']
                self.val = dataset['val']
            else:
                self.test = dataset['test']
            
            self.args_ret = dataset['args_ret']
            
            return True
        else:
            print('Dataset not found in',file_name) 
            return False

    def splitDataset(self, shuffle = True, split_size=0.70, preserve_lang=None, balance_by_class=None, balance_error=0.1, time_grouped_by_company=False):
        '''
        shuffle :  True for shuffling the data before split.
        split_size    : For training, by  default is 0.7, having a remaining 0.3 for validation. For testing: 1.0
        preserve_lang: Attribe to preserve the proportion when split 
        balance_by_class: if negative, data is not balanced but the total number the samples are obtained from the less frequent
        '''
        
        is_short = False
        group_size = 1

        if len(self.train.all_input_ids) == 0 and len(self.train.text) > 0:
            is_short = True
            dataset_size = len(self.train.text)
            if len(self.train.text_lang)>0: assert len(self.train.text) == len(self.train.text_lang)
        else:
            dataset_size = len(self.train.all_input_ids)
            group_size = int(len(self.train.all_input_ids) /len(self.train.text))
            #assert len(self.train.all_words) == len(self.train.all_labels) or  len(self.train.all_words) == len(self.train.all_metadata)*group_size
            assert len(self.train.all_words) == len(self.train.all_metadata) 
            assert len(self.train.all_metadata) == len(self.train.text) 
            assert len(self.train.all_labels) == len(self.train.all_label_ids)
            if self.train.all_input_ids is not None and len(self.train.all_input_ids)>=0:   
                assert len(self.train.all_input_ids) == len(self.train.all_tokens) 
                assert len(self.train.all_input_ids) == len(self.train.all_segment_ids)
        
        if balance_by_class is not None  and len(self.train.all_labels)>0 and len(self.train.idx2labels)==0:
            labels_dict = {item for item in self.train.all_labels}
            self.train.idx2labels = {i:item for i,item in enumerate(labels_dict)}
            self.train.labels2idx = {self.train.idx2labels[item]:item for item in self.train.idx2labels}
        
        if len(self.train.all_labels) != len(self.train.all_metadata):
            print('All labels size:', len(self.train.all_labels))
            print('All metadata size:', len(self.train.all_metadata))
            self.save_dataset(save_full_dataset=True)

        if balance_by_class is not None and len(self.train.all_labels)>0 and len(self.train.idx2labels)>0:
            number_classes = len(self.train.idx2labels)
            if len(self.train.all_metadata) >= np.abs(balance_by_class) and number_classes >0:  
                dataset_size = np.abs(balance_by_class)
                if time_grouped_by_company:
                    #Number of documents per class, only considering the last year document
                    temp_labels = [self.train.all_labels[i] for i,item_m in enumerate(self.train.all_metadata) if self.dataset_last_documents.get(item_m['company_id']) ==self.train.all_metadata[i]['document']]
                    c_records_per_class = Counter(temp_labels).most_common(number_classes) 
                else:
                    c_records_per_class = Counter(self.train.all_labels).most_common(number_classes)

                min_count_class = c_records_per_class[-1][1]
                print('Counter for balancing', c_records_per_class)
                
                if balance_by_class >0:
                    number_records_per_class = dataset_size / number_classes*(1-balance_error)
                    max_count_class = number_records_per_class * (1+balance_error)
                    if min_count_class < number_records_per_class:
                        print('Min required per class: ', min_count_class)
                        raise Exception("Is not possible to balance the data with ", number_classes, " classes and required", number_records_per_class," samples per class(e=",balance_error,"). Increase the dataset or decrease the number of records required.")
                    else:
                        print("Balancing the data with ", number_classes, "classes and", number_records_per_class, " (avg) samples per class, where the min is " , min_count_class," per class.")
                else:
                    total_data = sum([item[1] for item in c_records_per_class]) 
                    if total_data>= np.abs(balance_by_class):
                        number_records_per_class = min(int(dataset_size/number_classes)+1, min_count_class) #TODO if I specify 400, i want at the end 400*number of years
                        max_count_class = dataset_size
                    else:
                        msg = "Balancing the data with " + str(number_classes)  + " classes and " + str(total_data) + " total records, is not possible for the goal " + str(abs(balance_by_class))
                        raise Exception(msg)

            else:
                raise Exception("There is no enought data to reach the number of requested records(",balance_by_class,")", 'having in metadata', len(self.train.all_metadata))
            
            #reduce dataset based on number of records
            total_selected_indexes_per_class = {class_selected:0 for class_selected in self.train.labels2idx}
            indexes_list = []
            indexes_dict = {} 
            total_records = 0 #Balancing all the classes to the minimum
            for i_item, class_item in enumerate(self.train.all_labels): 
                if not time_grouped_by_company or self.dataset_last_documents.get(self.train.all_metadata[i_item]['company_id']) == self.train.all_metadata[i_item]['document']:
                    if total_selected_indexes_per_class[class_item] < number_records_per_class:
                        total_selected_indexes_per_class[class_item] += 1
                        indexes_list.append(i_item)
                        indexes_dict[i_item] = True 
                        total_records += 1
                    
                    if total_records == dataset_size:
                        break
            #Complete the classes with the remaining classes until reach the required dataset size
            if len(indexes_list)<dataset_size:
                for i2_item in range(0, len(self.train.all_labels)):
                    if not time_grouped_by_company or self.dataset_last_documents.get(self.train.all_metadata[i2_item]['company_id']) == self.train.all_metadata[i2_item]['document']:                     
                        class_item = self.train.all_labels[i2_item]
                        if indexes_dict.get(i2_item) is None and len(indexes_list) < dataset_size and total_selected_indexes_per_class[class_item] < max_count_class:
                            total_selected_indexes_per_class[class_item] += 1 
                            indexes_list.append(i2_item)
                            indexes_dict[i2_item]=True

            temp_new_indexes = {}
            self.number_years = 1
            if time_grouped_by_company: #look for all the other documents from the same company
                for index_item in indexes_dict: 
                    list_docs = [i for i,item in enumerate(self.train.all_metadata) if str(item['company_id'])==str(self.train.all_metadata[index_item]['company_id']) and i!=index_item]
                    if len(list_docs) >0: 
                        temp_new_indexes[self.train.all_metadata[index_item]['company_id']] = list_docs
                        self.number_years = len(list_docs) + 1

                dataset_yearly = {}  #add the other documents to a temporal dataset
                for company_id in temp_new_indexes:
                    metadata_list = []
                    for index_document in  temp_new_indexes[company_id]:
                        metadata_list.append([self.train.all_metadata[index_document]['year'], self.train.all_metadata[index_document]['document'],index_document])
                        metadata_list.sort()
                    
                    dataset_yearly[company_id] = {'text':[],'text_lang':[],'all_words':[],'all_labels':[],'all_metadata':[],'all_label_ids':[],'all_input_ids':[],'all_input_mask':[],'all_segment_ids':[],'all_tokens':[]}
                    for document_year in metadata_list: 
                        dataset_yearly[company_id]['text'].append(self.train.text[document_year[2]])
                        dataset_yearly[company_id]['text_lang'].append(self.train.text_lang[document_year[2]])
                        dataset_yearly[company_id]['all_words'].append(self.train.all_words[document_year[2]])
                        dataset_yearly[company_id]['all_labels'].append(self.train.all_labels[document_year[2]])
                        dataset_yearly[company_id]['all_metadata'].append(self.train.all_metadata[document_year[2]])
                        dataset_yearly[company_id]['all_label_ids'].append(self.train.all_label_ids[document_year[2]])
                        dataset_yearly[company_id]['all_tokens'].append(self.train.all_tokens[document_year[2]])
                        if len(self.train.all_input_ids)>0: 
                            start_group = document_year[2]*group_size
                            end_group = start_group+group_size
                            dataset_yearly[company_id]['all_input_ids'].extend(self.train.all_input_ids[start_group:end_group])
                            dataset_yearly[company_id]['all_input_mask'].extend(self.train.all_input_mask[start_group:end_group])
                            dataset_yearly[company_id]['all_segment_ids'].extend(self.train.all_segment_ids[start_group:end_group])

    
            dataset_size = len(indexes_list)
            self.train.text =  self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.text) 
            self.train.text_lang =  self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.text_lang) 
            self.train.all_words =  self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_words) 
            self.train.all_labels = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_labels) 
            self.train.all_metadata =self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_metadata)          
            self.train.all_label_ids = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_label_ids)     

            if len( self.train.all_input_ids)>0: 
                self.train.all_input_ids = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_input_ids, group_size=group_size) 
                self.train.all_input_mask = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_input_mask, group_size=group_size) 
                self.train.all_segment_ids = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_segment_ids, group_size=group_size)   
                self.train.all_tokens = self.select_from_indexes(list_indexes=indexes_list, _collection = self.train.all_tokens, group_size=group_size)  

            print('Dataset created with ',  dataset_size, 'records')

        #text_vector = self.train.text if len(self.train.text)>0 else self.train.text_lang
        if group_size is None or group_size == 1:
            total_split = int(dataset_size*split_size)
            total_val = dataset_size - total_split
        else:
            group_size_split = int(len(self.train.text)*split_size)  
            total_split = group_size_split*group_size
            total_val = len(self.train.text)*group_size- total_split 
            assert total_split + total_val == len(self.train.text)*group_size

        if self.is_train:
            print('Split Parameters: Trainable', dataset_size, ', Train:', group_size_split, 'Test:', (dataset_size-group_size_split))
        else:
            total_val = 0
            print('Split Parameters: Total', dataset_size, ', Test:', group_size_split)

        print('Dataset size:', dataset_size)
        
        if self.train.all_metadata is not None and len(self.train.all_metadata) >0:
            print('Number Documents:', len({item['document'] for item in self.train.all_metadata}))
            print('Number Pages:', len({item['document']+str(item['page']) for item in self.train.all_metadata}))
        
        indexes_vector = np.arange(len(self.train.text)) if  len(self.train.text) > 0 else np.arange(dataset_size)
        
        if self.is_train:
            self.val = Object() 
        else:
            self.test = Object()
        
        if preserve_lang and balance_by_class is None:
            #calculate size per language and split type 
            total_per_lang = Counter([item for item in self.train.text_lang])
            sizes_per_lang = {item:{} for item in total_per_lang}
            for lang in total_per_lang:
                _trainable = total_per_lang[lang] 
                if self.is_train:
                    sizes_per_lang[lang]['train'] = int(_trainable*split_size)
                    sizes_per_lang[lang]['val'] = _trainable - sizes_per_lang[lang]['train'] 
                else:
                    sizes_per_lang[lang]['test'] = int(total_per_lang[lang]*split_size)
             
            if self.is_train:
                lang_indexes_train = {item:[] for item in total_per_lang} 
                lang_indexes_val = {item:[] for item in total_per_lang}  
            else:
                lang_indexes_test = {item:[] for item in total_per_lang} 
            
            #get indexes per languages
            for _index in indexes_vector:
                lang = self.train.text_lang[_index]
                if self.is_train:
                    if len(lang_indexes_train[lang]) < sizes_per_lang[lang]['train']:
                        lang_indexes_train[lang].append(_index) 
                    elif len(lang_indexes_val[lang]) < sizes_per_lang[lang]['val']:
                        lang_indexes_val[lang].append(_index)          
                else:
                    if len(lang_indexes_test[lang]) < sizes_per_lang[lang]['test']:
                        lang_indexes_test[lang].append(_index) 
        
            split_indexes =[]
            if self.is_train: 
                val_indexes = []
                for lang in lang_indexes_train:
                    split_indexes.extend(lang_indexes_train[lang])
                    if total_val > 0: val_indexes.extend(lang_indexes_val[lang])
            else:
                for lang in lang_indexes_test:
                    split_indexes.extend(lang_indexes_test[lang])

            if shuffle:
                split_indexes = self.shuffle(only_indexes=True, list_indexes=split_indexes)
                if total_val > 0:  val_indexes = self.shuffle(only_indexes=True, list_indexes=val_indexes)
                
        else:
            if balance_by_class is not None:
                total_per_class = Counter([item for item in self.train.all_labels])
                sizes_per_class = {item:{} for item in total_per_class}
                
                for class_r in total_per_class:
                    _trainable = total_per_class[class_r] 

                    if self.is_train:
                        sizes_per_class[class_r]['train'] = int(_trainable*split_size)
                        sizes_per_class[class_r]['val'] = _trainable - sizes_per_class[class_r]['train'] 
                    else:
                        sizes_per_class[class_r]['test'] = int(total_per_class[class_r]*split_size)
                print(sizes_per_class)
                if self.is_train:
                    class_indexes_train = {item:[] for item in total_per_class} 
                    class_indexes_val = {item:[] for item in total_per_class}  
                else:
                    class_indexes_test = {item:[] for item in total_per_class} 
                
                if shuffle:
                    indexes_vector = self.shuffle(only_indexes=True, list_indexes=indexes_vector) 
                    
                for _index in indexes_vector:
                    class_r = self.train.all_labels[_index]
                    if self.is_train:
                        if len(class_indexes_train[class_r]) < sizes_per_class[class_r]['train']:
                            class_indexes_train[class_r].append(_index) 
                        elif len(class_indexes_val[class_r]) < sizes_per_class[class_r]['val']:
                            class_indexes_val[class_r].append(_index)          
                    else:
                        if len(class_indexes_test[class_r]) < sizes_per_class[class_r]['test']:
                            class_indexes_test[class_r].append(_index) 

                split_indexes =[]
                if total_val > 0: 
                    val_indexes = []
                    for class_r in class_indexes_val: 
                        val_indexes.extend(class_indexes_val[class_r])
                    val_indexes = self.shuffle(only_indexes=True, list_indexes=val_indexes) 
                class_indexes_temp = class_indexes_train if self.is_train else class_indexes_test
                labels_c = [len(class_indexes_temp[item]) for item in class_indexes_temp]
                for i_pos_list in range(max(labels_c)):        
                    for class_r in class_indexes_temp:
                        if i_pos_list < len(class_indexes_temp[class_r]) : split_indexes.append(class_indexes_temp[class_r][i_pos_list])
                
                split_indexes = self.shuffle(only_indexes=True, list_indexes=split_indexes) 
            else: #preserve lang and balance
                if self.is_train:
                    split_indexes = indexes_vector[:total_split]
                    if total_val > 0: val_indexes = indexes_vector[total_split:]
                else:
                    split_indexes = indexes_vector[:total_split]

        if is_short:
            if shuffle:
                if self.is_train:
                    if total_val>0:
                        print('Shuffling validation data...')
                        self.val.text = self.shuffle(only_indexes=False,list_indexes=val_indexes,_collection= copy.deepcopy(self.train.text))
                        self.val.text_lang = self.shuffle(only_indexes=False,list_indexes=val_indexes,_collection=copy.deepcopy(self.train.text_lang))
                        self.val.all_words = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection =copy.deepcopy(self.train.all_words))
                        self.val.all_labels = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_labels))
                        self.val.all_metadata = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_metadata))
                        

                    print('Shuffling training data...')
                    self.train.text = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text)
                    self.train.text_lang = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection=self.train.text_lang)  
                    self.train.all_words = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection =self.train.all_words)
                    self.train.all_labels = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_labels)
                    self.train.all_metadata = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_metadata) 
                    
                else:
                    print('Shuffling testing data...')
                    self.test.text = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text)
                    self.test.text_lang = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection=self.train.text_lang)  
                    self.test.all_words = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection =self.train.all_words)
                    self.test.all_labels = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_labels)
                    self.test.all_metadata = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_metadata)
                    
        else:
            if self.is_train: 
                if total_val > 0:
                    print('Shuffling validation data...')
                    self.val.text = self.shuffle(only_indexes=False,list_indexes=val_indexes,_collection= copy.deepcopy(self.train.text))
                    self.val.text_lang = self.shuffle(only_indexes=False,list_indexes=val_indexes,_collection= copy.deepcopy(self.train.text_lang))
                    self.val.all_words = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection =copy.deepcopy(self.train.all_words))
                    self.val.all_labels = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_labels))
                    self.val.all_metadata = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_metadata))

                    
                    if len(self.train.all_label_ids)>0: 
                        self.val.all_label_ids = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_label_ids))
                    else:
                        self.val.all_label_ids = []
                        
                    if self.train.all_input_ids is not None and len(self.train.all_input_ids)>0:
                        self.val.all_input_ids = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_input_ids),group_size=group_size) 
                        self.val.all_input_mask =  self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_input_mask),group_size=group_size)
                        self.val.all_segment_ids =  self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_segment_ids),group_size=group_size)
                        self.val.all_tokens = self.shuffle(only_indexes=False, list_indexes=val_indexes, _collection = copy.deepcopy(self.train.all_tokens),group_size=group_size) 
                    else:
                        self.val.all_input_ids = []
                        self.val.all_input_mask = []
                        self.val.all_segment_ids = []
                        self.val.all_tokens = []

                    print('Shuffling training data...') 
                    self.train.text = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text)
                    self.train.text_lang = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text_lang)
                    self.train.all_words = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_words) 
                    self.train.all_labels = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_labels) 
                    self.train.all_metadata = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_metadata) 

                    if len(self.train.all_label_ids)>0:
                        self.train.all_label_ids_no_shuffled = copy.deepcopy(self.train.all_label_ids)
                        self.train.all_label_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_label_ids) 
                    else:
                        self.train.all_label_ids = []

                    if self.train.all_input_ids is not None and len(self.train.all_input_ids)>0:
                        self.train.all_input_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_input_ids,group_size=group_size) 
                        self.train.all_input_mask = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_input_mask,group_size=group_size) 
                        self.train.all_segment_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_segment_ids,group_size=group_size) 
                        self.train.all_tokens = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_tokens,group_size=group_size) 
                    else:
                        self.train.all_input_ids = []
                        self.train.all_input_mask = []
                        self.train.all_segment_ids = []
                        self.train.all_tokens = []
            else:
                print('Shuffling testing data...')
                self.test.text = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text)
                self.test.text_lang = self.shuffle(only_indexes=False,list_indexes=split_indexes,_collection= self.train.text_lang)
                self.test.all_words =  self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_words) 
                self.test.all_labels = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_labels) 
                self.test.all_metadata =self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_metadata) 

                if len(self.train.all_label_ids)>0:
                    self.test.all_label_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_label_ids) 
                else:
                    self.test.all_label_ids = []

                if self.train.all_input_ids is not None and len(self.train.all_input_ids)>0:
                    self.test.all_input_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_input_ids,group_size=group_size) 
                    self.test.all_input_mask = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_input_mask,group_size=group_size) 
                    self.test.all_segment_ids = self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_segment_ids,group_size=group_size)
                    self.test.all_tokens =self.shuffle(only_indexes=False, list_indexes=split_indexes, _collection = self.train.all_tokens,group_size=group_size)   
            
        if  time_grouped_by_company: 
            if self.is_train: 
                self.val = self.add_yearly_data(dataset_yearly, self.val)

                data_train = Object()
                data_train.text, data_train.text_lang,data_train.all_words,data_train.all_metadata, data_train.all_tokens = copy.deepcopy(self.train.text), copy.deepcopy(self.train.text_lang), copy.deepcopy(self.train.all_words), copy.deepcopy(self.train.all_metadata), copy.deepcopy(self.train.all_tokens)
                data_train.all_input_ids, data_train.all_input_mask, data_train.all_segment_ids = copy.deepcopy(self.train.all_input_ids), copy.deepcopy(self.train.all_input_mask), copy.deepcopy(self.train.all_segment_ids)
                self.train = self.add_yearly_data(dataset_yearly, data_train, self_data=True) 
                data_train = None
            else:
                self.test = self.add_yearly_data(dataset_yearly, self.test)
            dataset_yearly = None

    def add_yearly_data(self, dataset_yearly, base_dataset, self_data=False):
        if len(base_dataset.all_metadata) > 0:
            group_size = int(len(base_dataset.all_input_ids)/len(base_dataset.all_metadata))
        else:
            raise Exception('Group size is zero. metadata list = 0')
        if self_data:
            final_data = self.train
            #total_metatada = len(self.train.all_metadata)
            #base_dataset.all_labels = []
            #if self.train.all_input_ids is not None:
            #    total_inputs = len(self.train.all_input_ids)
        else:
            final_data = Object()
            final_data.all_labels, final_data.all_label_ids = base_dataset.all_labels, base_dataset.all_label_ids
        final_data.text, final_data.text_lang, final_data.all_words,  final_data.all_metadata, final_data.all_tokens = [], [], [],[],[]
        final_data.all_input_ids, final_data.all_input_mask, final_data.all_segment_ids = [], [], []
        
        total_removed_companies = 0
        for i_company, company_metadata in enumerate(base_dataset.all_metadata):
            company_id = company_metadata['company_id'] 
            if dataset_yearly.get(company_id) is not None:
                years = {item['year']:0 for item in dataset_yearly.get(company_id)['all_metadata']}
                years.update({base_dataset.all_metadata[i_company]['year']:0})
                if len(years) ==len(dataset_yearly.get(company_id)['all_metadata'])+1:
                    final_data.text.extend(dataset_yearly[company_id]['text'])
                    final_data.text.append(base_dataset.text[i_company])
                    final_data.text_lang.extend(dataset_yearly[company_id]['text_lang'])
                    final_data.text_lang.append(base_dataset.text_lang[i_company])
                    final_data.all_words.extend(dataset_yearly[company_id]['all_words'])
                    final_data.all_words.append(base_dataset.all_words[i_company])
                    final_data.all_tokens.extend(dataset_yearly[company_id]['all_tokens'])
                    final_data.all_tokens.append(base_dataset.all_tokens[i_company])
                    #final_data.all_labels.extend(dataset_yearly[company_id]['all_labels'])
                    #final_data.all_labels.append(base_dataset.all_labels[i_company])
                    final_data.all_metadata.extend(dataset_yearly[company_id]['all_metadata'])
                    final_data.all_metadata.append(base_dataset.all_metadata[i_company])
                    #final_data.all_label_ids.extend(dataset_yearly[company_id]['all_label_ids'])
                    #final_data.all_label_ids.append(base_dataset.all_label_ids[i_company])
                    
                    if base_dataset.all_input_ids is not None and len(base_dataset.all_input_ids)>0:
                        start_group = i_company*group_size
                        end_group = start_group + group_size 
                        final_data.all_input_ids.extend(dataset_yearly[company_id]['all_input_ids'])
                        final_data.all_input_ids.extend(base_dataset.all_input_ids[start_group:end_group])
                        final_data.all_input_mask.extend(dataset_yearly[company_id]['all_input_mask'])
                        final_data.all_input_mask.extend(base_dataset.all_input_mask[start_group:end_group])
                        final_data.all_segment_ids.extend(dataset_yearly[company_id]['all_segment_ids'])
                        final_data.all_segment_ids.extend(base_dataset.all_segment_ids[start_group:end_group])
                else:
                    total_removed_companies +=1
            else:
                total_removed_companies += 1
            '''         
            if self_data:
            self.train.text = final_data.text[total_metatada:]
            self.train.text_lang = final_data.text_lang[total_metatada:]
            self.train.all_words = final_data.all_words[total_metatada:]
            #self.train.all_labels = final_data.all_labels[total_metatada:]
            self.train.all_metadata = final_data.all_metadata[total_metatada:]
            #self.train.all_label_ids = final_data.all_label_ids[total_metatada:]
            if base_dataset.all_input_ids is not None and len(final_data.all_input_ids)>0:
                self.train.all_input_ids = final_data.all_input_ids[total_inputs:]
                self.train.all_input_mask = final_data.all_input_mask[total_inputs:]
                self.train.all_segment_ids = final_data.all_segment_ids[total_inputs:] 
            '''
        subtext = 'train' if self_data else 'val'
        print('Total companies removed because there were not files with the required number of years', total_removed_companies, 'in', subtext)
        return final_data

    def select_from_indexes(self, list_indexes=None, _collection=None, group_size=1):        
        response_collection = []
        #adding new values of each record at the end of the collection  
        for _index in list_indexes:
            if len(_collection)>0:
                if group_size ==1:
                    response_collection.append(_collection[_index])
                else:
                    response_collection.extend(_collection[_index*group_size:(_index+1)*group_size])

        return response_collection
            

    def shuffle(self, only_indexes = False, list_indexes=None, _collection=None, group_size=1, parallel_groups=1):
        if list_indexes is None:
            list_indexes = np.arange(len(_collection)) 
        if only_indexes:
            final_vector = []
            total_records = len(list_indexes)
            step = int(total_records / parallel_groups)
            sum_total = sum(list_indexes)
            for shuffle_group in range(0,total_records, step):
                shuffled_indexes = list_indexes[shuffle_group:shuffle_group+step]
                random.shuffle(shuffled_indexes)
                final_vector.extend(shuffled_indexes)
            
            assert sum(final_vector) == sum_total

            return final_vector
            
        else:
            response_collection = self.select_from_indexes(list_indexes=list_indexes, _collection=_collection, group_size=group_size)

            return response_collection
            

    def filter_dataset(self, doc_filters):
        dataset_final_t0 = []
        if doc_filters.get('lang') is not None:
            print('Filtering by lang', doc_filters['lang'])
            for document in self.dataset:
                if doc_filters.get('lang') == document['language']:
                    dataset_final_t0.append(document) 
            doc_filters.pop('lang')

            print('After language Filter', Counter([item.get('risk_desc') for item in dataset_final_t0 if item.get('risk_desc') is not None]))
            print('Number companies', len({item.get('company_id') for item in dataset_final_t0}))
            print('Number documents', len({item.get('file_name') for item in dataset_final_t0}))
            
        else:
            dataset_final_t0 = self.dataset
        
        
        dataset_final_t1 = []
        if doc_filters.get('type_doc') is not None:
            print('Filtering by type of document', doc_filters['type_doc'])
            for document in dataset_final_t0:
                if document.get('document_type') is not None:
                    if doc_filters.get('type_doc') in document['document_type']:
                        dataset_final_t1.append(document)
            doc_filters.pop('type_doc')
            del dataset_final_t0
            
            print('After language by type of document', Counter([item.get('risk_desc') for item in dataset_final_t1 if item.get('risk_desc') is not None]))
            print('Number companies', len({item.get('company_id') for item in dataset_final_t1}))
            print('Number documents', len({item.get('file_name') for item in dataset_final_t1}))
        else:
            dataset_final_t1 = dataset_final_t0

        dataset_final_t2 = []
        if doc_filters.get('ind_cia') is not None:
            print('Filtering by industry', doc_filters['ind_cia'])
            for document in dataset_final_t1:
                if document.get('company_industry') is not None:
                    if doc_filters.get('ind_cia') == document['company_industry']:
                        dataset_final_t2.append(document)
            doc_filters.pop('ind_cia')            
            del dataset_final_t1

            print('After language by industry', Counter([item.get('risk_desc') for item in dataset_final_t2 if item.get('risk_desc') is not None]))
            print('Number companies', len({item.get('company_id') for item in dataset_final_t2}))
            print('Number documents', len({item.get('file_name') for item in dataset_final_t2}))
        else:
            dataset_final_t2 = dataset_final_t1

        dataset_final_t3 = []
        filtered = False
        if len(doc_filters) >0:
            for filter_ in doc_filters:
                if  filter_ !='last_doc':
                    print('Filtering by other filter', filter_, doc_filters[filter_])
                    filtered = True
                    for document in dataset_final_t2:
                        if type(doc_filters[filter_]) ==list:
                            filter_list = doc_filters[filter_]
                        else:
                            filter_list = [doc_filters[filter_]]

                        for s_filter in filter_list:
                            if document[filter_] == s_filter:
                                dataset_final_t3.append(document) 
                                break
            
                print('After language by other filter', filter_, Counter([item.get('risk_desc') for item in dataset_final_t3 if item.get('risk_desc') is not None]))
                print('Number companies', len({item.get('company_id') for item in dataset_final_t3}))
                print('Number documents', len({item.get('file_name') for item in dataset_final_t3}))

        if not filtered:
            dataset_final_t3 = dataset_final_t2
            
        dataset_final_t4 = []
        dataset_final_t5 = []
        self.dataset_last_documents = {}
        companies_per_year = {}
        documents_loaded = {}
        if doc_filters.get('last_doc') is not None:
            print('Filtering by last document')
            company_dic_last_year = {}
            for document in dataset_final_t3: #get the last document's year per company
                if document.get('company_id') is not None:
                    company_id = str(document['company_id'])
                    company_dic_last_year[company_id] =  document['document_year'] if company_dic_last_year.get(company_id) is None else max(document['document_year'], company_dic_last_year[company_id])
            for document in dataset_final_t3: #get all the documents that falls into the period requested time period
                if document.get('document_year') is not None and len(str(document.get('document_year')))>0: 
                    if int(document['document_year']) > int(company_dic_last_year[str(document['company_id'])]) + doc_filters.get('last_doc'):
                        dataset_final_t4.append(document)
                        if documents_loaded.get(document['file_name']) is None: #ensure document was not previously registered and add counter for number of documents per company
                            companies_per_year[str(document['company_id'])] = 1 if companies_per_year.get(str(document['company_id'])) is None else  companies_per_year[str(document['company_id'])] + 1
                            documents_loaded[document['file_name']]=True
                    if self.dataset_last_documents.get(str(document['company_id'])) is None: #add the list of cias with its last document 
                        if int(document['document_year']) == int(company_dic_last_year[str(document['company_id'])]):
                            self.dataset_last_documents[str(document['company_id'])] = document['file_name']
            if abs(doc_filters.get('last_doc')) > 1:
                for company_id in companies_per_year:
                    if companies_per_year[company_id] != abs(doc_filters.get('last_doc')):
                        self.dataset_last_documents.pop(company_id)
                
                companies_per_year = {item:companies_per_year[item] for item in companies_per_year if companies_per_year[item] ==abs(doc_filters.get('last_doc'))}
                #filter out companies and their documents which does not have complete number of required documents.
                for document_record in dataset_final_t4:
                    company_id = document_record['company_id']
                    if companies_per_year.get(company_id) is not None:
                        dataset_final_t5.append(document_record) 
            else:
                dataset_final_t5 = dataset_final_t4

            doc_filters.pop('last_doc')  

            
            print('After last document', filter_, Counter([item.get('risk_desc') for item in dataset_final_t5 if item.get('risk_desc') is not None]))
            print('Number companies', len({item.get('company_id') for item in dataset_final_t5}))
            print('Number documents', len({item.get('file_name') for item in dataset_final_t5}))

        else:
            dataset_final_t5 = dataset_final_t3
            
        self.dataset = dataset_final_t5
        return dataset_final_t5

    def create_dict_from_datatext(self, data, types_list=None, debugnote=None):
        print(debugnote)
        if len(data) == 0 or type(data[0])==dict:
            return data
        dataset = []
        headers= {i:item.replace(' ','_').lower() for i,item in enumerate(data[0][:-1].split('\t'))}
        if len(types_list) < len(headers): types_list = types_list + [str for item in range(len(headers)-len(types_list))]
        ini_len_header = len(headers)
        multiheader_index =[item for  item in headers if len(headers[item].split(','))>1]
        add_new_headers = False
        if len(multiheader_index) >0:
            add_new_headers = True
            attached_headers =  {i+ini_len_header:header for i, header in enumerate(headers[multiheader_index[0]].split(','))}
            headers.update(attached_headers)
        
        print('Total rows ', len(data))
        
        for row in tqdm.tqdm(range(1, len(data))): 
            data_vec = data[row][:-1].split('\t')
            if headers[0]!=data_vec[0]:
                info = {headers[i]:types_list[i](data_vec[i]) for i,item in enumerate(data_vec)}
                if len(info)>0:
                    if add_new_headers:
                        new_data = {headers[i+ini_len_header]:types_list[-1](item) for i, item in enumerate(info[headers[multiheader_index[0]]].split(','))}
                        info.update(new_data)
                        info.pop(headers[multiheader_index[0]])
                    dataset.append(info)
        return dataset

    def merge_dataset(self, dataset, dataset_tmp, level='record'):
        if level == 'record':
            for i_record, record in enumerate(dataset):
                dataset[i_record].update(dataset_tmp[i_record])
        elif level=='page':
            pages_not_found = {}
            dataset_tmp_dict = {item['file_name'] + '_' + str(item['page_number']):item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                page_id = record['file_name'] + '_' + str(record['page_number'])
                if dataset_tmp_dict.get(page_id) is not None:
                    dataset[i_record].update(dataset_tmp_dict[page_id])
                else:
                    pages_not_found[record['file_name'] + "_" + str(record['page_number'])] = 0 
            del dataset_tmp_dict
    
            if len(pages_not_found) >0:
                print('Documents not found: ', list(pages_not_found.keys()))
    
        elif level=='doc':
            docs_not_found = {}
            dataset_tmp_dict = {item['file_name']:item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                if dataset_tmp_dict.get(record['file_name']) is not None:
                    temp_record = dataset_tmp_dict[record['file_name']]
                    if temp_record.get('page_number') is not None:
                        temp_record.pop('page_number')
                    dataset[i_record].update(temp_record)
                else:
                    docs_not_found[record['file_name']] = 0
            del dataset_tmp_dict
    
            if len(docs_not_found)>0:
                print('Documents not found: ', list(docs_not_found.keys()))
    
        elif level=='cia':
            companies_not_found = {}
            documents_without_companies= {}
            dataset_tmp_dict = {int(item['company_id']):item for item in dataset_tmp}
            for i_record, record in enumerate(dataset):
                if record.get('company_id') is not None:
                    if dataset_tmp_dict.get(record['company_id']) is not None:
                            dataset[i_record].update(dataset_tmp_dict[record['company_id']])
                    else:
                        companies_not_found[record['company_id']] = 0 
                else:
                    documents_without_companies[record['file_name']] = 0
            del dataset_tmp_dict

            if len(companies_not_found)>0:
                print('Companies not found: ', len(list(companies_not_found.keys())))
                print('Documents without companies: ', len(list(documents_without_companies.keys())))
    
        return dataset

    def getTextListByDocument(self, dataset_perc=1, group_by_page= False):
        doc_list = {item['file_name']:0 for item in self.dataset}
        dataset_size = int(len(doc_list) * min(1,dataset_perc))
        documents_detail= {}
        company_number = 0
        row_number =0
        while len(documents_detail) < dataset_size and company_number < dataset_size:
            current_row = self.dataset[row_number]
            file_name = current_row['file_name']
            page_number = current_row['page_number']
            if documents_detail.get(file_name) is None:                 
                documents_detail[file_name] = {} if group_by_page else []
                company_number +=1 
            if group_by_page:
                if documents_detail[file_name].get(page_number) is None:
                    documents_detail[file_name][page_number] = []
                documents_detail[file_name][page_number].append(current_row)
            else:
                documents_detail[file_name].append(current_row)
            row_number += 1

        return documents_detail


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default=None,
        type=str,
        required=True,
        help="Action to execute. Options: extract, process_links, load", )
    
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The data dir where the pdf files are located.", )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The data dir where the output files are going to be saved.", )

    parser.add_argument(
        "--metadata_file",
        default=None,
        type=str,
        required=False,
        help="If there is metadata information at document level, the file name should be specified.", )

    parser.add_argument(
        "--split_type",
        default="line",
        type=str,
        required=False,
        help="Level of granularity for generation of each dataset record. Line or Word.", )

    parser.add_argument(
        "--detect_lang",
        default=False,
        type=bool,
        required=False,
        help="True if the generator is going to detect the language of each page. Otherwise, lang parameter is going to be considered.", )
    
    parser.add_argument(
        "--lang",
        default='fra',
        type=str,
        required=False,
        help="To avoid language detection. Options: fra, eng, deu, swe.", )
    
    parser.add_argument(
        "--total_workers",
        default=14,
        type=int,
        required=False, 
        help="For parallel processing, if the number is bigger than the number of processors available, the maximum number of processor are going to be considered.", )

    parser.add_argument(
        "--worker_load",
        default=14,
        type=int,
        required=False, 
        help="For parallel processing, if the number is bigger than the number of processors available, the maximum number of processor are going to be considered.", )

    parser.add_argument(
        "--filter_last_doc",
        default=None,
        type=bool,
        required=False, 
        help="In Loading, only for considering the last document of the company.", )

    parser.add_argument(
        "--filter_type_doc",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document of the selected type.", )
    
    parser.add_argument(
        "--filter_industry_cia",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document's company of the selected industry.", )
    
    parser.add_argument(
        "--filter_lang",
        default=None,
        type=str,
        required=False, 
        help="In Loading, only for considering the document in the specified language.", )

    parser.add_argument(
        "--max_pages",
        default=30,
        type=int,
        required=False, 
        help="Number of maximum number of files to be processed.", )

    parser.add_argument(
        "--max_docs",
        default=10000,
        type=int,
        required=False, 
        help="Number of maximum number of files to be processed.", )

    parser.add_argument(
        "--distrib_tool",
        default="ray",
        type=str,
        required=False, 
        help="The tool for redistributing the parallel work. Ray or pool.", )
        
    parser.add_argument(
        "--filetype",
        default="json",
        type=str,
        required=False, 
        help="Extension for reading/generating the files (json, txt).", )
    
    
    parser.add_argument(
        "--multifile_option",
        default="batch",
        type=str,
        required=False, 
        help="When batch: each processor/task is going to create a file with the corresponding data. If document, one file per document and if page one file per page. When none, one single file.", )
    
    parser.add_argument(
        "--filter_list_path",
        default=None,
        type=str,
        required=False, 
        help="List of documents to process.", )

    args = parser.parse_args()

    print(args)
    print('Cuda available ',torch.cuda.is_available())
    print('Number of available GPUS ', torch.cuda.device_count())
    print('Number of processors ' + str(mp.cpu_count()))

    if args.action == 'extract':
        oFDExTGenerator = FDExTGenerator(args.data_dir, args.output_dir, args.split_type, args.filetype, args.filter_list_path,args.multifile_option)
        oFDExTGenerator.generateFiles(args.detect_lang, args.lang, args.worker_load, args.total_workers, args.max_docs, args.max_pages, args.distrib_tool, from_links=False) 
    elif args.action == 'process_links':
        oFDExTGenerator = FDExTGenerator(args.data_dir, args.output_dir, args.split_type, args.filetype, args.filter_list_path,args.multifile_option)
        oFDExTGenerator.generateFiles(args.detect_lang, args.lang, args.worker_load, args.total_workers, args.max_docs, args.max_pages, args.distrib_tool, from_links=True) 
    elif args.action == 'process_file':
        oFDExTGenerator = FDExTGenerator(args.data_dir, args.output_dir, args.split_type, args.filetype, filter_list_path = None, multifile_option = args.multifile_option, check_loaded_docs =False)
        oFDExTGenerator.generateSingleFile(args.data_dir, args.detect_lang, args.lang, max_pages_per_time=args.max_pages)  
    elif args.action =='load':
        oFDExt = FDExt(args.data_dir, args.output_dir)
        oFDExt.loadDataset(filter_last_doc=args.filter_last_doc, filter_type_doc=args.filter_type_doc, filter_industry_cia=args.filter_industry_cia , filter_lang=args.filter_lang)
        print()