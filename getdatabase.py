from sqlalchemy import create_engine
import pandas as pd

class GetDatabase:
    def __init__(self, username, password, host, port, database):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.engine = None
        self.connect()

    def connect(self):
        try:
            db_url = f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.engine = create_engine(db_url)
            print("Veritabanına başarıyla bağlanıldı!")
        except Exception as e:
            print("Bağlantı hatası:", e)

    def fetch_data(self, table_name):
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
            print(f"{table_name} tablosundan veri çekildi:")
            return df
        except Exception as e:
            print(f"Veri çekme hatası: {e}")

    def empty_data(self,table_name):
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        print("Eksik veri kontrolü:")
        print(df.isnull().sum())

