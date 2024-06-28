import argparse
import random
from nlpde import FDExt
from sentimentModel import runSentimentModel 

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
        default=1,
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
        default=3,
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
        default="page_type",
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
        "--dense_model_type",
        default=0,
        type=int,
        required=False, )
   
    parser.add_argument(
        "--max_files",
        default=None,
        type=int,
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
        "--run_id",
        default=None,
        type=str,
        required=False, 
        help="Run Id to log in the metadata store", )
    
    parser.add_argument(
        "--filter_type_doc",
        default=None,
        type=str,
        required=False, 
        help="", )
    
    parser.add_argument(
        "--filter_last_doc",
        default=None,
        type=int,
        required=False, 
        help="", )
    
    
    parser.add_argument(
        "--trim_type",
        default="start",
        type=str,
        required=False, 
        help="", )
    
    args = parser.parse_args()    

    print('Args', args)

    time_grouped_by_company = False
    
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
        print('Starting') 
        raw_loaded = oDataset.loadDataset(filter_last_doc=args.filter_last_doc, filter_type_doc=args.filter_type_doc,  additional_filters = None, perc_data=args.perc_data, labels_path=args.labels_path, data_sel_position=data_sel_position, filter_lang=None,max_number_files=args.max_files) 
        
        if raw_loaded:            
            print('Preparing')
            clean_text_list =  ['The notes in the annex form an integral part of the annual accounts',
                      'Les notes figurant en annexe font partie intégrante des comptes annuels',
                      'Die Anhänge sind integraler Bestandteil der Jahresabschlüsse',
                      'notes', 'annexes', 'annexe', 'annex', 'document émis électroniquement','Anhang',
                      'verkürzte', 'abridged', 'abrégé','mention',
                      'bilan', 'balance sheet', 'bilanz',
                      'profits et pertes', 'profit and loss', 'gewinn', 'verlustrechnung'] 
                   
            oDataset.prepare_input_data(model_dir=args.output_dir, model_type =args.model_type, group_level='page', y_label_type=args.y_label_type, labels = args.labels, y_top_n=args.y_top_n, trim_type=args.trim_type,tokenizer=None,sequence_strategy=args.sequence_strategy, sequence_shift=args.sequence_shift, max_segments_bert=args.max_segments_bert, terms_filter_path=None, clean_data=args.clean_data, remove_numbers=args.remove_numbers,time_grouped_by_company=time_grouped_by_company, normalization_column=None, worker_workload=args.worker_workload,distibution_tool=args.distrib_tool, max_proportion_numbers_words=0,clean_text_list=clean_text_list)
            oDataset.splitDataset(shuffle = True, split_size=split_size, preserve_lang=True, balance_by_class=args.balance_by_class, balance_error=args.balance_error,time_grouped_by_company=time_grouped_by_company)
            oDataset.save_dataset()
        else:
            seq_l = oDataset.args_ret['lstm_sequence_length']
            print(f'Initial sequence lenght from dataset: {seq_l}')
            print(f'Task argument: {args.max_segments_bert}, trim_type: {args.trim_type}')
            if seq_l != args.max_segments_bert:
                if seq_l > args.max_segments_bert and args.max_segments_bert ==1 and args.trim_type =='random':
                    oDataset.args_ret['lstm_sequence_length'] =1
                    oDataset.args_ret['is_bert_lstm'] = False
                
                selection_objects = [oDataset.train, oDataset.val]
                for dataset_type in selection_objects:
                    selection_vector = []
                    for i in range(0,len(dataset_type.all_input_ids),seq_l):
                        rand_index = random.randint(0,seq_l-1)
                        if dataset_type.all_input_ids[i+rand_index][0] ==0:
                            rand_index=0
                        selection_vector_group = [False for item in range(seq_l)]
                        selection_vector_group[rand_index] = True
                        selection_vector.extend(selection_vector_group)
                    
                    #apply segment selection to the corresponding objects
                    temp_all_tokens = []
                    temp_all_segment_ids = []
                    temp_all_input_mask = []
                    temp_all_input_ids = []

                    for i, is_segment in enumerate(selection_vector):
                        if is_segment:
                            temp_all_tokens.append(dataset_type.all_tokens[i])
                            temp_all_segment_ids.append(dataset_type.all_segment_ids[i])
                            temp_all_input_mask.append(dataset_type.all_input_mask[i])
                            temp_all_input_ids.append(dataset_type.all_input_ids[i])
                    
                    dataset_type.all_tokens = temp_all_tokens
                    dataset_type.all_segment_ids = temp_all_segment_ids
                    dataset_type.all_input_mask = temp_all_input_mask
                    dataset_type.all_input_ids = temp_all_input_ids
                    
            else:
                raise Exception(f'Dataset specified ({seq_l}) not coincide with the number of requested segments ({args.max_segments_bert}).')
        if args.action =='train_generate_data' or args.action == 'test_generate_data': 
            return    
        print('Finalized preparing data.')
        
        runSentimentModel(dataset=oDataset,action=args.action,data_dir=args.data_dir,output_dir=args.output_dir,learning_rate=args.learning_rate,
                        batch_size=args.batch_size, n_epochs=args.n_epochs, is_bert_lstm=oDataset.args_ret['is_bert_lstm'], 
                    lstm_sequence_length=oDataset.args_ret['lstm_sequence_length'], lstm_hidden_layers = args.lstm_hidden_layers, lstm_stacked_layers=oDataset.args_ret['lstm_stacked_layers'],
                    fast_break=args.fast_break, model_type = args.model_type, tokenizer = oDataset.args_ret['tokenizer'], debug_gpu_memory=args.debug_gpu_memory, optimizer=args.optimizer, 
                    dropout_perc=args.dropout_perc, number_layers_freeze = args.number_layers_freeze,local_model_location=args.model_dir, dense_model_type=args.dense_model_type, run_id=args.run_id)

if __name__ =='__main__':
    run()