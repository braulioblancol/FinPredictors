import argparse
from DBConnector import SQLConnector
import _queries
import math
import json
import os
from nlpde import FDExt, FDExTGenerator
from pathlib import Path

def writeCompanyLabelForBankruptcy2(output_dir,dataset):  
    # get document label for bankruptcy
    document_content = 'company_id\trisk_desc\n'
    for i_company, company_obj in enumerate(dataset):  
        file_name = company_obj['filename']
        date_aa = company_obj['annual_accounts_from']
        company_id = company_obj['company_id']
        if company_id is None:
            risk_label='No Risk'
        else:
            risk_label='Risk'
        document_content += file_name + '\t' + risk_label + '\n'
    
    #writing file
    document_path = os.path.join(output_dir,'text_doc_labelRisk.txt')     
    with open(document_path,'w',encoding='utf-8') as f:
        f.writelines(document_content)
        print('List of ', len(dataset),' write in ', document_path)

def writeCompanyLabelForBankruptcy(output_dir,dataset):      
    # get document label for bankruptcy
    document_content = 'company_id\trisk_desc\tdeletion_date\tregistration_date\tfiling_date\n'
    companies_loaded = {}
    for i_company, company_obj in enumerate(dataset):  
        if companies_loaded.get(company_obj['company_id']) is None:

            bankruptcy = company_obj['bankrupcty_flag']  # Risk Level = High
            deleted = company_obj['deleted_flag']  # Risk Level = Medium
            voluntary = company_obj['voluntary_flag']  # Risk Level = Medium
            observed = company_obj['observed_flag']  # Risk Level = High, Very High if with deleted
            faillite = company_obj['faillite_flag']  # Risk Level = High, Very High if with deleted
            risk_desc = ''
            if bankruptcy ==1 and voluntary == 0: # bankruptcy companies except voluntary closed companies 
                risk_desc = 'Risk'
            elif observed + faillite + deleted + bankruptcy + voluntary == 0:
                risk_desc = 'No Risk'
            else:
                risk_desc = 'Not sure'
                
            companies_loaded[company_obj['company_id']] = {}
            
            dataset[i_company]['risk_desc'] = risk_desc
            deletion_time = dataset[i_company]['deletion_time']
            deletion_time = deletion_time[:10] if deletion_time is not None else ''
            filing_date_parsed = dataset[i_company]['filing_date_parsed']
            filing_date_parsed = filing_date_parsed[:10] if filing_date_parsed is not None else ''
            registration_date = dataset[i_company]['registration_date'] 
            registration_date = registration_date[:10] if registration_date is not None else ''
            document_content += str(dataset[i_company]['company_id'])  + '\t' + dataset[i_company]['risk_desc'] + "\t" + deletion_time + "\t" + registration_date + "\t" + filing_date_parsed + '\n'
    
    #writing file
    document_path = os.path.join(output_dir,'text_cia_labelRisk.txt')     
    with open(document_path,'w',encoding='utf-8') as f:
        f.writelines(document_content)
        print('List of ', len(companies_loaded),' write in ', document_path)

def writeListDocumentsPerCia(output_dir,dataset):   
    document_content = 'company_id\tmax_year\tdocument_id\tdocument_name\n'
    dataset_docs = {}
    for i_company, company_obj in enumerate(dataset):  
        company_id = company_obj['company_id']   
        max_aa = company_obj['max_aa']   
        file_name = company_obj['filename']   
        doc_aa = company_obj['doc_aa']   
        if dataset_docs.get(company_id) is None: dataset_docs[company_id] = []
        dataset_docs[company_id].append([doc_aa,file_name,max_aa])
        document_content += str(company_id)  + '\t' + str(max_aa) + "\t" + file_name + "\t" + str(doc_aa) + '\n'
    
    #writing file
    document_path = os.path.join(output_dir,'text_cia_documents.txt')     
    with open(document_path,'w',encoding='utf-8') as f:
        f.writelines(document_content)
        print('List of ', len(dataset),' write in ', document_path)
 
    number_years = 3
    reduced_dataset = {}
    for company_id in dataset_docs:
        if reduced_dataset.get(company_id) is None: reduced_dataset[company_id] = []
        for document in dataset_docs[company_id]:
            if document[2]-number_years < document[0]:
                reduced_dataset[company_id].append(document) 
    
    final_dataset = {}
    for company_id in reduced_dataset:
        if len(reduced_dataset[company_id]) ==number_years: 
            final_dataset[company_id] = reduced_dataset[company_id]

    document_content_threeyears = 'company_id\tmax_year\t' 
    for i_company, company_id in enumerate(final_dataset):  
        if i_company ==0: document_content_threeyears += '\t'.join([item+str(i) for i,item in enumerate(['file_name','document_year']*len(final_dataset[company_id]))]) + '\n'
        company_doc_list = final_dataset[company_id] 
        document_content_threeyears += str(company_id)  + '\t' + str(company_doc_list[0][2]) +'\t' +'\t'.join(['\t'.join([str(item[1]),str(item[0])]) for item in company_doc_list]) + '\n'

    document_path_threeyears = os.path.join(output_dir,'text_cia_documents_3y.txt')     
    with open(document_path_threeyears,'w',encoding='utf-8') as f:
        f.writelines(document_content_threeyears)
        print('List of ', len(document_content_threeyears),' write in ', document_path_threeyears)

    print(len(final_dataset))
    print('')

def checkDocumentGap(data_dir, file_path, output_dir, files_dir, filter_list_path, number_years=3):
    oFDExTGenerator = FDExTGenerator(args.data_dir, args.output_dir, 'line', 'json', "D:\\datasets\\LBR\\documents_scope_filter.json",'batch')
    local_docs = oFDExTGenerator.loaded_docs
    print('Documents in local dataset', len(local_docs))
    
    #DOCUMENTS IN YOBA (3 YEARS)
    with open(file_path, 'r') as f: 
        line_set = f.readlines()

    header = True
    companies_yoba = {}
    documents_yoba = {}
    for line in line_set:
        line = line.split('\t')
        if not header:
            companies_yoba[int(line[0])] = {0:[line[2],int(line[3])],1:[line[4],int(line[5])],2:[line[6],int(line[7].split("\n")[0])]}
            documents_yoba[line[2]] = line[0]
            documents_yoba[line[4]] = line[0]
            documents_yoba[line[6]] = line[0]
        else:
            header = False
    gap_documents = {item:int(documents_yoba[item]) for item in documents_yoba if local_docs.get(item) is None}
    
    with open(os.path.join(data_dir,'document_gap.json'), 'w')as f:
        json.dump(gap_documents, f)

    content_list = [Path(os.path.join(data_dir,file_name)) for file_name in os.listdir(data_dir) if 'text_line_page_info' in file_name]
    local_dataset = []
    for file in content_list:
        with open(file, 'r') as jf:
            datafile = json.load(jf)
            local_dataset.extend(datafile)

    documents_local = {item['file_name']:0 for item in local_dataset} 
    print('Documents in yoba not in local dataset',len(gap_documents))
    
    if filter_list_path is not None:
        with open(filter_list_path, 'r') as f:
            filter_files =  json.load(f)

        docs_scope = {item:0 for item in filter_files}

    print('Documents in Yoba, not in filter', len([item for item in documents_yoba  if docs_scope.get(item) is None]))
    print('Documents in scope', len(docs_scope))

    #oDataset = FDExt(data_dir, output_dir)
    #oDataset.loadDataset(filter_last_doc=-3, filter_type_doc='eCDF', additional_filters={'page_type':['Unknown','']}, perc_data=1)

    print('Documents in yoba, not in local', len([item for item in documents_yoba  if documents_local.get(item) is None])) 

    
    ''' companies_local = {item['company_id']:[] for item in oDataset_full.page_info if item.get('company_id') is not None}
    companies_years_temp = {str(item['company_id'])+"_"+str(item.get('document_year')):0 for item in oDataset_full.page_info if item.get('company_id') is not None}
    for company_year in companies_years_temp:
        company_id = int(company_year.split("_")[0])
        year = int(company_year.split("_")[1])
        companies_local[company_id].append(year)
    
    documents_local = {}
    for company_id in companies_local: 
        max_year = max(companies_local[company_id])
        total_docs = 0
        for year in companies_local[company_id]:
            if year > max_year - number_years:
                total_docs +=1
                if total_docs == number_years:
                    documents_local[company_id] = total_docs
  
    
    companies_in_yoba_not_in_local = [company_id for company_id in companies_yoba if documents_local.get(company_id) is None] 
    companies_in_both_yoba = [company_id for company_id in companies_yoba if documents_local.get(company_id) is not None]
    companies_in_local_not_yoba = [company_id for company_id in documents_local if companies_yoba.get(company_id) is None]

    with open(os.path.join('D:\\datasets\\LBR\\original', 'workingDocumentsList.json'), 'r')as f:
        documents_in_scope = json.load(f)

    files_in_dir_and_yoba = [filename[:-4] for filename in os.listdir(files_dir) if 'pdf' in filename and documents_yoba.get(filename[:-4]) is not None]
    print('Number of file in Yoba and pdf',len(files_in_dir_and_yoba))
    full_ocr_docs = {item['file_name']:0 for item in oDataset_full.page_info}
            
    docs_not_ocr = [item for item in files_in_dir_and_yoba if full_ocr_docs.get(item) is None]
    print('Number of docs not processed OCR', len(docs_not_ocr)) '''

    print('')

def splitDocumentLinks(file_path, links_per_file, output_dir):
    link_list = []
    with open(file_path,'r') as f:
        link_list = f.readlines()
        link_list = [item.split('\n')[0] for item in link_list]

    total_files = int(len(link_list)/links_per_file) + 1

    for i_file in range(total_files):
        output_file = os.path.join(output_dir,Path(file_path).stem + str(i_file).rjust(5, '0') + ".txt")
        file_list = link_list[i_file*links_per_file:(i_file+1)*links_per_file]
        if len(file_list) >0:
            with open(output_file,'w') as f: 
                f.writelines('\n'.join(file_list))
            print('Saving file', output_file)

    print('Total generated files', total_files)

def generateDocumentLinks(links_per_file, document_type_filter, output_dir):
    oSQLConnector = SQLConnector()
    results = oSQLConnector.executeQuery(_queries.getDocumentLinksPerDocType(document_type_filter))
    number_files = math.ceil(len(results)/links_per_file)
    list_links = [item['link'] for item in results if not item['link'] is None]
    i_start = 0

    for n in range(number_files):
        output_file = os.path.join(output_dir, "document_links_" + document_type_filter + "_" + str(n).rjust(5, '0') + ".txt")
        with open(output_file,"w") as f:
            f.write('\n'.join(list_links[i_start:i_start + links_per_file]))
        i_start += links_per_file
        print('Saved file: ' + output_file)

def generateNormalizationAmountsPerCia(output_dir):
    oSQLConnector = SQLConnector()
    results = oSQLConnector.executeQuery(_queries.getNormalizationAmountPerCompany())
    output_file = os.path.join(output_dir,'text_cia_normalizationAmount.txt')   
    header = "company_id\tnormalization\n"

    with open(output_file,"w") as f:
        f.write(header + '\n'.join([str(item['company_id'])+"\t"+str(math.ceil(item['amount'])) for item in results]))

    print('Saved file: ' + output_file)

def generatePageType(data_dir, output_dir,perc_data,parallel_workers,distibution_tool,worker_workload):  
    oDataset = FDExt(data_dir, output_dir, action='train', total_records=None,dataset_name=None)
    oDataset.loadDataset(filter_last_doc=None, filter_type_doc='eCDF',  additional_filters = None, perc_data=perc_data, labels_path=None, data_sel_position='first', filter_lang=None,max_number_files=None)
    oDataset.get_grouped_text(rows_list=oDataset.dataset, clean_data=False, remove_numbers=False, group_level='page', y_label_type=None, filter_min_words=None, additional_headers = None,parallel_workers=parallel_workers, distrib_tool=distibution_tool, workload=worker_workload)
    labels = []
    lang_formats = []
    document_dict = {item['document']:None for item in oDataset.train.all_metadata}
    for i, page_text in enumerate (oDataset.train.all_words):
        document = oDataset.train.all_metadata[i]['document']
        if document_dict.get(document) is None:
            document_dict[document] = i
            page_set = [oDataset.train.all_words[i] for i,item in enumerate(oDataset.train.all_metadata) if item['document']==document]
            full_text = ' '.join(page_set)
            last_label = 'unknown'
            lang_fin_statement = check_financial_statement(full_text) 
            if lang_fin_statement is not None:          
                for page_text in page_set:   
                    lang_formats.append(lang_fin_statement)
                    if len(full_text.strip())<10:
                        labels.append('blank')
                        continue 
                    lang_page_financial_statement = check_financial_statement(page_text) 
                    if lang_page_financial_statement is not None: #'6lb40D, 35m0mX
                        if 'bilan' in page_text.replace('é','e').replace('á','a') or \
                            'balance sheet'  in page_text or \
                             'bilanz' in page_text:
                            last_label = 'balance_sheet'
                        elif 'profits et pertes' in page_text.replace('é','e').replace('á','a') or \
                            'profit and loss'  in page_text or \
                             ('gewinn'  in page_text and 'verlustrechnung'  in page_text): 
                            last_label = 'profit_and_loss'
                        elif last_label =='unknown':
                            last_label = 'unknown_financial_statement'
                    else:
                        if last_label != 'unknown':
                            last_label = 'annex'
                    labels.append(last_label)
            else:
                for page_text in page_set:
                    labels.append('')
                    lang_formats.append(lang_fin_statement)
        
    assert len(labels)==len(oDataset.train.all_words)
    assert len(labels)==len(lang_formats)

    output_file = os.path.join(output_dir, "text_line_page_type_aa2.txt")
    with open(output_file,"w") as f:
        final_text = "file_name\tpage_number\tpage_type_aa\tlang_format_aa\n"
        for i,label in enumerate(labels):
            metadata_info = oDataset.train.all_metadata[i]
            lang_fs = lang_formats[i] if lang_formats[i] is not None else ''
            final_text += str(metadata_info['document']) + '\t' + str(metadata_info['page']) + '\t' + str(label) + '\t' + lang_fs + '\n'
        f.write(final_text)

    print('Saved file: ' + output_file)


            

def check_financial_statement(text):
    if 'Les notes figurant en annexe font partie intégrante des comptes annuels'.lower()  in text:
        return 'fra'
    if 'The notes in the annex form an integral part of the annual accounts'.lower()  in text:
        return 'eng'
    if 'Die Anhänge sind integraler Bestandteil der Jahresabschlüsse'.lower() in text:
        return 'deu'
    return None

def generateLinksGapDocs(docs_gap_path, output_dir,batch_size = 5000):
    
    with open(docs_gap_path, 'r') as f:
        gap_docs = json.load(f)
    list_docs = [item for item in gap_docs]
    number_queries = int(len(gap_docs)/batch_size +1)

    oSQLConnector = SQLConnector()
    returning_list = []
    for i_file in range(number_queries):
        list_docs_query = list_docs[i_file*batch_size:(i_file+1)*batch_size]
        result_link = oSQLConnector.executeQuery(_queries.getDocumentLinkPerFileName(list_docs_query))
        returning_list.extend([item['link'] for item in result_link])
    
    output_file = os.path.join(output_dir, "document_links_gap.txt")
    with open(output_file,"w") as f:
        f.write('\n'.join(returning_list))

    print('Saved file: ' + output_file)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default=None,
        type=str,
        required=True,
        help="Action to execute. Options: generate, load ", )
    
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The data dir where the pdf files are located.", )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The data dir where the output files are going to be saved.", )

    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        required=False,
        help=".", )

    parser.add_argument(
        "--files_dir",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--filter_list_path",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--links_per_file",
        default=None,
        type=int,
        required=False, )

    parser.add_argument(
        "--document_type_filter",
        default=None,
        type=str,
        required=False, )
    
    parser.add_argument(
        "--parallel_workers",
        default=1,
        type=int,
        required=False, )    

    parser.add_argument(
        "--distibution_tool",
        default='mp',
        type=str,
        required=False, )

    
    parser.add_argument(
        "--worker_workload",
        default=1,
        type=int,
        required=False, )  
    
    parser.add_argument(
        "--perc_data",
        default=1,
        type=float,
        required=False, )
      
    args = parser.parse_args()

    if args.action =='risk_label_cia':
        oSQLConnector = SQLConnector()
        results = oSQLConnector.executeQuery(_queries.getBankruptcyDataByDocument()) 
        writeCompanyLabelForBankruptcy(args.output_dir, results)    
    #if args.action =='risk_label_cia':
    #    oSQLConnector = SQLConnector()
    #    results = oSQLConnector.executeQuery(_queries.getBankruptcyDataByCompany())
    #    writeCompanyLabelForBankruptcy(args.output_dir, results)   
    if args.action =='doc_cias':
        oSQLConnector = SQLConnector()
        results = oSQLConnector.executeQuery(_queries.getDocumentsPerCompany())
        writeListDocumentsPerCia(args.output_dir, results) 
    elif args.action =='docgaps_links':  
        generateLinksGapDocs(args.data_dir, args.output_dir)  
    elif args.action == 'check_doc_gap':
        checkDocumentGap(args.data_dir, args.file_path, args.output_dir, args.files_dir, args.filter_list_path)
    elif args.action == 'genDocLinks':
        generateDocumentLinks(args.links_per_file, args.document_type_filter, args.output_dir)
    elif args.action == 'splitFile':
        splitDocumentLinks(args.data_dir, args.links_per_file, args.output_dir)
    elif args.action =='genDocNoramlize':
        generateNormalizationAmountsPerCia(args.output_dir)
    elif args.action =='genPageTypeAA':
        generatePageType(args.data_dir, args.output_dir, args.perc_data,args.parallel_workers,args.distibution_tool,args.worker_workload)