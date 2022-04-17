import pandas as pd


class Parser:
    def __init__(self, filename):
        self.filename = filename

    def get_data(self):
        data = pd.read_csv(self.filename, sep=';', encoding='cp1251')
        data = data.apply(lambda x: x.str.replace(',', '.'))
        headers = data.values[0, 2:].tolist()
        headers[headers.index('Pлин')] = 'Рлин'

        d = {}
        for i, header in enumerate(headers):
            if header in d:
                d[header] += 1
                headers[i] = header + '_' + str(d[header])
            else:
                d[header] = 1

        data = data.iloc[2:, 2:]
        data.columns = headers
        data = data.reset_index(drop=True)

        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        for i in range(0, data.shape[0]):
            if (pd.isnull(data['КГФ'][i])) & (pd.notnull(data['КГФ_2'][i])):
                data['КГФ'][i] = data['КГФ_2'][i] * 1000

        data.drop('КГФ_2', axis=1, inplace=True)
        data.dropna(how='all', subset=['КГФ', 'G_total'], inplace=True)
        data = data.reset_index(drop=True)
        return data
