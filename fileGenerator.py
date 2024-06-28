import argparse
from DBConnector import SQLConnector,PostgresConnector
import _queries 
import pandas as pd
import csv


def get_list_detailed(base_dic, key):
    new_list = base_dic.get(key)
    if new_list  is not None: new_list = ' - '.join([item + " (" + new_list[item] +")" for item in new_list])
    return new_list

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        default="Excel",
        type=str,
        required=False,
        help="If the results are required or the list of documents.", )

    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=False,
        help="The file name for saving the results.", )

    parser.add_argument(
        "--account_number",
        default=None,
        type=str,
        required=False,
        help="Column name in the database ", )
    
    parser.add_argument(
        "--min_amount",
        default=5000000000,
        type=float,
        required=False,
        help="Minimum amount for filtering.", )

      
    args = parser.parse_args()
 
    oDBConnector = SQLConnector()
    results = oDBConnector.executeQuery(_queries.getCompaniesFilteredByAccountMinAmount()) 
    if args.action =='Excel':
        dfData = pd.DataFrame.from_dict(results)
        dfData.to_excel(args.output_file)
    elif args.action =='txt':
        document_list = [str(item['company_id']) for item in results]
        results = oDBConnector.executeQuery(_queries.getAllDocumentLinksForCompanies(document_list)) 
        with open(args.output_file,'w') as f:
            f.writelines('\n'.join([item['link'] for item in results if item['link'] is not None]))
    elif args.action =='audited_companies':    
        oDBConnector = PostgresConnector()
        results = oDBConnector.executeQuery(_queries.getFlagDocumentsForFilter('auditor'))
        company_list = {item['company_id']:{} for item in results}
        for result in results:
            if company_list[result['company_id']].get(result['type_info']) is None:
                company_list[result['company_id']][result['type_info']] = {}
            company_list[result['company_id']][result['type_info']][result['filename'].strip()] = result['pages']
        
        result_dict = {'company_id':[], 'auditor':[],'auditor_enterprise_approved':[],'auditor_internal_comissaire':[],'auditor_approved':[],'auditor_enterprise':[]}

        for company_id in company_list: 
            result_dict['company_id'].append(company_id)
            result_dict['auditor'].append(get_list_detailed(company_list[company_id], 'auditor'))
            result_dict['auditor_enterprise'].append(get_list_detailed(company_list[company_id], 'auditor_enterprise'))
            result_dict['auditor_enterprise_approved'].append(get_list_detailed(company_list[company_id], 'auditor_enterprise_approved'))
            result_dict['auditor_approved'].append(get_list_detailed(company_list[company_id], 'auditor_approved'))
            result_dict['auditor_internal_comissaire'].append(get_list_detailed(company_list[company_id], 'auditor_internal_comissaire'))
            
        dfData = pd.DataFrame.from_dict(result_dict)
        dfData.to_excel(args.output_file)

        print('')
    