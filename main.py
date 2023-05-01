import argparse
from nlpde import FDExt
import numpy as np
import os
from sentimentModel import runSentimentModel, runROCAnalysis 
from CustomExplainer import CustomBERTExplainer
import json
from pathlib import Path

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
        "--model_dir",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
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
        "--perc_data",
        default=0.15,
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
        default=400,
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
        default=500,
        type=int,
        required=False, )     
                 
    parser.add_argument(
        "--split_size",
        default=0.70,
        type=float,
        required=False, )     
    
    parser.add_argument(
        "--max_segments_bert",
        default=10,
        type=int,
        required=False, )     
    
    parser.add_argument(
        "--labels_path",
        default=None,
        type=str,
        required=False, )     
     
    parser.add_argument(
        "--debug_gpu_memory",
        default=False,
        type=bool,
        required=False, )  

    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        required=False, )     
    
    parser.add_argument(
        "--y_label_type",
        default="risk_desc",
        type=str,
        required=False, )     
    
    
    parser.add_argument(
        "--labels",
        default=None,
        type=str,
        required=False, 
        nargs="*")     

    parser.add_argument(
        "--dropout_perc",
        default=0.1,
        type=float,
        required=False, )    
        
    parser.add_argument(
        "--title",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--tokenizer",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--balance_by_class",
        default=None,
        type=int,
        required=False, )

    parser.add_argument(
        "--number_layers_freeze",
        default=10,
        type=int,
        required=False, )

    parser.add_argument(
        "--data_sel_position",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--balance_error",
        default=0.1,
        type=float,
        required=False, )
        
    parser.add_argument(
        "--terms_list",
        default=None,
        type=str,
        required=False, )
        
    parser.add_argument(
        "--clean_data",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--remove_numbers",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--filter_lang",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--dense_model_type",
        default=0,
        type=int,
        required=False, )

    parser.add_argument(
        "--filter_last_doc",
        default=-1,
        type=int,
        required=False, )
    
    parser.add_argument(
        "--max_files",
        default=None,
        type=int,
        required=False, )

    parser.add_argument(
        "--normalization_column",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--distrib_tool",
        default=None,
        type=str,
        required=False, 
        help="The tool for redistributing the parallel work. Ray or pool.", )

    parser.add_argument(
        "--worker_workload",
        default=10,
        type=int,
        required=False, 
        help="Number of documents to work in parallel per worker.", )

    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=False, 
        help="Dataset already generated to load.", )

    parser.add_argument(
        "--only_annexes",
        default=False,
        type=bool,
        required=False, 
        help="If only consider financial annexes", )
    
    parser.add_argument(
        "--max_samples",
        default=10,
        type=int,
        required=False, )
        
    parser.add_argument(
        "--total_steps",
        default=2,
        type=int,
        required=False, )

    parser.add_argument(
        "--max_significant_words",
        default=10,
        type=int,
        required=False, )
        
    args = parser.parse_args()    

    print('Args', args)

    time_grouped_by_company = False
    if abs(args.filter_last_doc) >1:
        time_grouped_by_company = True
    
    action=args.action.split("_")[0]
    if action == 'train' or action=='test':
        oDataset = FDExt(args.data_dir, args.output_dir, action, total_records=args.balance_by_class,dataset_name=args.dataset_name)
        if args.action == 'test':
            if  args.data_sel_position is None: 
                data_sel_position='last'
            else:
                data_sel_position=args.data_sel_position

            split_size =  1
        else:   
            data_sel_position='first'
            split_size =  args.split_size

        print('Sel position of data:', data_sel_position)
        if args.only_annexes:
            additional_filters = {'page_type':'Unknown'}
        else:
            additional_filters = None
        print('Starting') #,
        raw_loaded = oDataset.loadDataset(filter_last_doc=args.filter_last_doc, filter_type_doc='eCDF',  additional_filters = additional_filters, perc_data=args.perc_data, labels_path=args.labels_path, data_sel_position=data_sel_position, filter_lang=args.filter_lang,max_number_files=args.max_files) 
        
        if raw_loaded:            
            print('Preparing')
            oDataset.prepare_input_data(model_dir=args.output_dir, model_type =args.model_type, group_level=None, y_label_type=args.y_label_type, labels = args.labels, y_top_n=args.y_top_n, trim_begining=False,tokenizer=args.tokenizer,sequence_strategy=args.sequence_strategy, sequence_shift=args.sequence_shift, max_segments_bert=args.max_segments_bert, terms_filter_path=args.terms_list, clean_data=args.clean_data, remove_numbers=args.remove_numbers,time_grouped_by_company=time_grouped_by_company, normalization_column=args.normalization_column, worker_workload=args.worker_workload,distibution_tool=args.distrib_tool)
            oDataset.splitDataset(shuffle = True, split_size=split_size, preserve_lang=True, balance_by_class=args.balance_by_class, balance_error=args.balance_error,time_grouped_by_company=time_grouped_by_company)
            oDataset.save_dataset()
                    
        if args.action =='train_generate_data' or args.action == 'test_generate_data': 
            '''
            dataset_path = oDataset.get_dataset_name()
            total_rows = 0
            with open(dataset_path, 'wb') as f:
                out_dataset = {'train':{},'val':{}}
                total_rows += len(oDataset.all_words) + len(oDataset.val.all_words)
                out_dataset['train']['words'] = oDataset.all_words
                out_dataset['train']['labels'] = oDataset.all_labels
                out_dataset['val']['words'] = oDataset.val.all_words
                out_dataset['val']['labels'] = oDataset.val.all_labels
                f.write(json.dumps(out_dataset).encode("utf-8"))
                print('Saved dataset in', dataset_path, 'Total Rows: ', total_rows)
            '''
            return    
        print('Finalized preparing data.')
        
        runSentimentModel(dataset=oDataset,action=args.action,data_dir=args.data_dir,output_dir=args.output_dir,learning_rate=args.learning_rate,
                        batch_size=args.batch_size, n_epochs=args.n_epochs, is_bert_lstm=oDataset.args_ret['is_bert_lstm'], 
                    lstm_sequence_length=oDataset.args_ret['lstm_sequence_length'], lstm_hidden_layers = args.lstm_hidden_layers, lstm_stacked_layers=oDataset.args_ret['lstm_stacked_layers'],
                    fast_break=args.fast_break, model_type = args.model_type, tokenizer = oDataset.args_ret['tokenizer'], debug_gpu_memory=args.debug_gpu_memory, optimizer=args.optimizer, 
                    dropout_perc=args.dropout_perc, number_layers_freeze = args.number_layers_freeze,local_model_location=args.model_dir, dense_model_type=args.dense_model_type)

    elif args.action == 'roc':
        runROCAnalysis(args.data_dir, args.output_dir, title=args.title)

    elif args.action == 'explain_input':
        oExplainer = CustomBERTExplainer(model=args.model_dir, tokenizer="bert-base-multilingual-uncased",type_model='custom')
        oExplainer.explain_model_input(args.data_dir, args.output_dir, max_samples=args.max_samples, total_steps=args.total_steps, dataset_name=args.dataset_name,max_significant_words=args.max_significant_words)

    elif args.action == 'explain_layer':
        oExplainer =  CustomBERTExplainer(model=args.model_dir, tokenizer="bert-base-multilingual-uncased",type_model='custom')
        oExplainer.explain_model_layers(args.data_dir, args.model_dir, args.output_dir,args.filter_last_doc,args.labels_path,args.perc_data,args.filter_lang,args.model_type,args.y_label_type,args.y_top_n,args.tokenizer,args.sequence_strategy,args.remove_stopwords,args.sequence_shift,args.max_segments_bert,args.terms_list,args.clean_data,args.remove_numbers,time_grouped_by_company, max_samples=10, total_steps=20)
    
    elif args.action == 'measure_models':
        datamodels_list = []
        with open(args.data_dir,'r') as f:
            datamodels_list=f.readlines()
        datamodels_list = [item.split("\n")[0] for item in datamodels_list]
        results_dic = {}
        for model_dir in datamodels_list:
            try:
                print(model_dir)
                oExplainer =  CustomBERTExplainer(model=model_dir, tokenizer="bert-base-multilingual-uncased",type_model='custom', is_train=True, number_layers_freeze=10)
                pytorch_total_params = sum(p.numel() for p in oExplainer.model.parameters())
                pytorch_total_train_params = sum(p.numel() for p in oExplainer.model.parameters() if p.requires_grad)
                print('Total number of parameters 10^6', int(pytorch_total_params/10000)/100)
                print('Total number of trainable parameters 10^6', int(pytorch_total_train_params/10000)/100)
                pytorch_size = sum(p.numel() for p in oExplainer.model.parameters()) * 4 / (1024 ** 2)
                print('Total size of model in MB', pytorch_size)
                oExplainer = None
                oPathfile = Path(os.path.join(model_dir,'pytorch_model.bin')).stat()
                size_disk = oPathfile.st_size/1024/1024
                results_dic[model_dir] = {'parameters':pytorch_total_params, 'trainable_parameters':pytorch_total_train_params, 'python_sizeMB':pytorch_size, 'disk_sizeMB':size_disk}
            except Exception as ex:
                print(ex)
        print('')
        print(results_dic)
            
if __name__ =='__main__':
    print('Running BERT models')
    run()