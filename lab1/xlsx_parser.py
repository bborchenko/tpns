import pandas as pd


class Parser:
    def __init__(self, file_name):
        self.file_name = file_name
        self.kick = []

    def parse(self):
        df = pd.read_excel(self.file_name, sheet_name='VU')
        print(df)
        self.merge_columns(df)

    def merge_columns(self, df):

        for index, row in df.iterrows():
            if str(row['Unnamed: 32']) != 'nan' and str(row['Unnamed: 33']) == 'nan':
                df['Unnamed: 33'][index] = df['Unnamed: 32'][index] / 1000

            if str(row['Unnamed: 31']) == 'nan' and str(row['Unnamed: 33']) == 'nan':
                self.kick.append(index)

        df.drop(index=self.kick, inplace=True)
        print(df)
