from pathlib import Path
import os
import pickle
import timeit
import argparse 
import json
from tqdm import tqdm

def getListFiles(content_list, suffixes_list = None, name_filter =None, filtered_dict=None):
    return [oFile for oFile in content_list if oFile.suffix in suffixes_list and (filtered_dict is None or filtered_dict.get(oFile.stem) is None) and (name_filter is None or name_filter in oFile.stem)]

def exploreDirectory(data_dir, suffixes_list = [".pdf",".png",".jpg","jpeg"], filtered_dict=None, name_filter=None, include_subdir=True):
    content_list = [Path(file_name) for file_name in os.listdir(data_dir)]
    #check if the main directory contains directories
    directory_list = [oFile for oFile in os.listdir(data_dir) if not os.path.isfile(oFile)]
    if len(directory_list) ==0 or not include_subdir:
        file_list = getListFiles(content_list, suffixes_list=suffixes_list, filtered_dict=filtered_dict, name_filter=name_filter)
    else:
        file_list = []
        for root, subdirs, files_ in os.walk(data_dir):
            content_list = [Path(os.path.join(root,file_name)) for file_name in files_]
            files = getListFiles(content_list, suffixes_list=suffixes_list, filtered_dict=filtered_dict, name_filter=name_filter)
            if len(files)>0:
                file_list.extend(files)
    return file_list

def run():
    start = timeit.default_timer()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False, 
        help="List of documents to process.", )

    parser.add_argument(
        "--output_filepath",
        default=None,
        type=str,
        required=False, 
        help="Name and path of the outputfile.", )
    
    parser.add_argument(
        "--filter_type_doc",
        default=None,
        type=str,
        required=False, 
        nargs="*",
        help="Type of document to search.", )
    
    parser.add_argument(
        "--text_search_list",
        default=None,
        type=str,
        required=False, 
        help="List of documents to process.", )
    
    parser.add_argument(
        "--set_to_search",
        default=None,
        type=str,
        required=False, 
        help="Name of the set to search in text_search_list.", )
    
    parser.add_argument(
        "--metadata_list",
        default=None,
        type=str,
        required=False, 
        help="List with the main metadata information about the documents.", )
    
    args = parser.parse_args()
    print(args)

    #load searchable list
    if Path(args.text_search_list).suffix ==".json":
        with open(args.text_search_list, 'rb') as f:
            dataset_to_search = pickle.load(f)
    else:
        with open(args.text_search_list, 'r', encoding='utf-8') as f:
            dataset_to_search = f.readlines()
            dataset_to_search = [item.replace("\n"," ").strip() for item in dataset_to_search]
            

    if args.set_to_search is not None:
        if args.set_to_search =='auditors:':
            list_to_search = [dataset_to_search[args.set_to_search][item]['auditor'] for item in dataset_to_search[args.set_to_search]]
        else:
            list_to_search = [item.replace(' S.à r.l.','').replace('S.à r.l','').replace('S.A.','').strip() for item in dataset_to_search[args.set_to_search]]

    else:
        list_to_search = [item for item in dataset_to_search]
    
    #load metadata file
    file_path = args.output_filepath

    #iterate file by file
    response_dataset = []
    directory_list = os.listdir(args.data_dir)
    if len(directory_list) == 0:
        file_list = exploreDirectory(args.data_dir,suffixes_list=['.json'], name_filter='raw_correct', include_subdir=False)        
    else:
        #look  for files in the main directory
        file_list = exploreDirectory(args.data_dir,suffixes_list=['.json'], name_filter='raw_correct', include_subdir=False) 
        if len(file_list)>0:
            print('Reading base directory', args.data_dir)
            temp_dataset = search_terms(file_list, list_to_search)
            response_dataset.extend(temp_dataset)
            save(file_path=file_path, response_dataset=response_dataset)
        #look into subdirectories        
        for i,current_dir in enumerate(directory_list):
            explore_dir = os.path.join(args.data_dir,current_dir)
            print('Reading directory:', explore_dir, i+1,"/",len(directory_list))
            file_list = exploreDirectory(explore_dir,suffixes_list=['.json'], name_filter='raw_correct', include_subdir=True)
            temp_dataset = search_terms(file_list, list_to_search)
            response_dataset.extend(temp_dataset)
            save(file_path=file_path, response_dataset=response_dataset)
    
    end = timeit.default_timer()-start
    print('Total minutes', end/60)

def search_terms(file_list, list_to_search):
    start = timeit.default_timer()
    response_dataset = []
    for file_location in tqdm(file_list): 
        file_name = file_location.stem.split("_")[-1]
        with open(file_location, 'r', encoding='utf-8') as f: #search by page, return page numbers where it was found
            _data = json.load(f)
              
        pages_with_info = {}
        for itemset_search in list_to_search:
            item_search_list = itemset_search.split(";")
            base_search = item_search_list[0]
            for item_search in item_search_list:
                for i_page, page_text in enumerate(_data['pages']):
                    if item_search.lower() in page_text.lower():
                        pages_with_info[str(i_page+1)] = 0
        
            if len(pages_with_info) >0:
                response_dataset.append({'file_name':file_name, 'search':base_search,'pages':[item for item in pages_with_info]})
            
    end = timeit.default_timer()-start
    print('--Total minutes', end/60)

    return response_dataset

def save(file_path,response_dataset):
    if Path(file_path).suffix == '.txt':
        report_content = [item['file_name'] + '\t' + item['search'] +'\t' + ','.join(item['pages']) + '\n' for item in response_dataset]
        header_content = 'file_name\tauditor\tpages\n' 
        with open(file_path,'w', encoding='utf-8') as f:
            f.write(header_content)
            f.writelines(report_content)
    else:
        with open(file_path,'w', encoding='utf-8') as f:
            json.dump(response_dataset)
    print('saved document in ',file_path)
    print('saved ', len(report_content), 'records')

if __name__=='__main__':
    run()