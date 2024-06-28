import uvicorn 
import numpy as np
from nlpde import FDExt
from fastapi import FastAPI
from sentimentModel import CustomBERTModel
import argparse 
import time

app = FastAPI(
    title="Document Type predictor Model API",
    description="API for predicting the document type of a certain page in a financial document",
    version="0.1",
) 

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_path",
    default=None,
    type=str,
    required=True, )

parser.add_argument(
    "--model_name",
    default="bert-base-multilingual-uncased",
    type=str,
    required=False, )

parser.add_argument(
    "--port",
    default="5100",
    type=str,
    required=False, )

args = parser.parse_args()    

print('Args')
print(str(args))  

port = args.port
#tokenizer_pred_mod_page_type = "bert-base-multilingual-uncased"
#model_pred_mod_page_type_dir = "D:\\models_HPC\\models\\PTPM\\ptpm_fr8_m0_s3" #"D:\\datasets\\LBR\\models\\CL\\PTMod"         
model_pred_mod_page_type = CustomBERTModel().from_pretrained(model_dir=args.model_path)

print(model_pred_mod_page_type.args['idx2labels'])

oDataset = FDExt()
tokenizer = oDataset.get_tokenizer(args.model_name)
padtypes = oDataset.get_pad_types(tokenizer)   

#PYDEVD_WARN_EVALUATION_TIMEOUT
@app.get("/predict-modif-page_type")
def predict_page_type_modification(page_text):
    time_start = time.time()
    oRecord = oDataset.prepareRecord(page_text, tokenizer)        
    _ ,input_ids_sample,token_types_sample,attention_masks_sample, _, _ = oDataset.segment_text_sample(oRecord['words'][0], oRecord['encoded'][0] , 
                                                                                                       trim_type='start', max_sequence_length=512, 
                                                                                                       max_segments_bert=model_pred_mod_page_type.lstm_sequence_length, 
                                                                                                       sequence_shift=400, padtypes=padtypes, return_tensors=True)    
                
    outputs = model_pred_mod_page_type(input_ids=input_ids_sample, attention_mask=attention_masks_sample, token_type_ids=token_types_sample, labels=None) 
    answer_index = int(outputs['logits'].detach().argmax())
    logits = outputs['logits'].detach().tolist()
    prediction = model_pred_mod_page_type.args['idx2labels'][answer_index]
    logits_temp = (logits - np.min(logits) ) 
    output_probability =np.max(logits_temp / np.sum(logits_temp))
    
    time_elapsed = (time.time() - time_start) 
    
    result = {"prediction": prediction, "probability": output_probability,'response_time':time_elapsed}
    return result 

if __name__ == '__main__':
    host = "0.0.0.0"
    print(f'Starting server on host {host} and port {port}')
    uvicorn.run(app, host=host, port=port) 