import pyodbc
import os
import _parameters

class SQLConnector:
    def __init__(self):
        connectionString = 'Driver={SQL Server};'
        connectionString += 'Server=%s;' % _parameters.SQL_SERVER
        connectionString += 'Database=%s;' % _parameters.SQL_DATABASE
        connectionString += 'UID=%s;' % _parameters.SQL_USER
        connectionString += 'PWD=%s;' % os.getenv('SQL_DBPASSWORD')
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
