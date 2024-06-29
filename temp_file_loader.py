import pickle 
import os
from sqlalchemy import text, create_engine

#with open("D:\\database\\train_dataset_s3.pickle",'rb') as f:
#    dataset = pickle.load(f)


def get_postgres_conn():
    pwd = os.environ['POSTGRES_PASSWORD']
    uid = os.environ['POSTGRES_USER']
    server = os.environ['POSTGRES_SERVER']
    port = 5432
    db = os.environ['POSTGRES_DB']
    try:
        connection_string = f'postgresql://{uid}:{pwd}@{server}:{port}/{db}' 
        print(f'postgres server:{server}, user: {uid}, pwd:{pwd}, connection string {connection_string}')
        cs = create_engine(connection_string)
        return cs
    except Exception as ex:
        raise ex
    
alchemy_engine = get_postgres_conn()  
with alchemy_engine.begin() as conn:
    q_select = "select document_id, text, service_response, o_json from documents_info_extraction_log "

    results = conn.execute(text(q_select))
    vector_results = []
    for row in results: 
        vector_results.append(record)


print('')