import pyodbc
import os
import _parameters
from sqlalchemy import create_engine, text

class SQLConnector:
    def __init__(self):
        connectionString = 'Driver={SQL Server};'
        connectionString += 'Server=%s;' % _parameters.SQL_SERVER
        connectionString += 'Database=%s;' % _parameters.SQL_DATABASE
        connectionString += 'UID=%s;' % _parameters.SQL_USER
        connectionString += 'PWD=%s;' % _parameters.SQL_PASSWORD
        connectionString += 'Trusted_Connection=no;'
        print(connectionString)
        self.connection = pyodbc.connect(connectionString)
        self.cursor = self.connection.cursor()


    def executeQuery(self, query):
        self.cursor.execute(query)
        companyList = []
        for row in self.cursor:
            new_row = {}
            for i, column in enumerate(row):
                new_row[row.cursor_description[i][0]] = column
            companyList.append(new_row)
        print('Query executed correctly. Number of returned rows: ' + str(len(companyList)))
            #companyList.append([column for column in row])
        return companyList


class PostgresConnector:
    
    def get_postgres_conn():
        pwd = _parameters.POSTGRES_PASSWORD
        uid = _parameters.POSTGRES_USER
        server = _parameters.POSTGRES_SERVER
        port = _parameters.POSTGRES_PORT
        db = _parameters.POSTGRES_DATABASE
        try:
            connection_string = f'postgresql://{uid}:{pwd}@{server}:{port}/{db}' 
            print(f'postgres server:{server}, user: {uid}, pwd:{pwd}, connection string {connection_string}')
            cs = create_engine(connection_string)
            return cs
        except Exception as ex:
            raise ex
        
    def executeQuery(self, query):
        alchemy_engine = PostgresConnector.get_postgres_conn()
        with alchemy_engine.connect() as conn:
            result = conn.execute(text(query)) 
             
        result_list = []
        table_header = {i:item for i,item in enumerate(list(result._metadata.keys))}
        for row in result:
            new_row = {}
            for i, column in enumerate(row):
                new_row[table_header[i]] = column
            result_list.append(new_row)  
        return result_list