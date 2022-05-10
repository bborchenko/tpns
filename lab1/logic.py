import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


class Logic:
    def __init__(self, data):
        self.data = data

    def get_properties(self):
        columns = self.data.columns
        d = {
            'Кол-во': [],
            '% пропусков': [],
            'Минимум': [],
            'Максимум': [],
            'Среднее': [],
            'Мощность': [],
            '% уникальных': [],
            'Первый квартиль(0.25)': [],
            'Медиана': [],
            'Третий квартиль(0.75)': [],
            'Стандартное отклонение': []
        }

        for h in self.data.columns:
            d['Кол-во'].append(self.data[h].count())
            d['% пропусков'].append(self.data[h].isna().sum() / len(self.data) * 100)
            d['Минимум'].append(self.data[h].min())
            d['Максимум'].append(self.data[h].max())
            d['Среднее'].append(self.data[h].mean())
            d['Мощность'].append(self.data[h].nunique())
            d['% уникальных'].append(self.data[h].nunique() / self.data[h].count() * 100)
            d['Первый квартиль(0.25)'].append(self.data[h].quantile(0.25))
            d['Медиана'].append(self.data[h].median())
            d['Третий квартиль(0.75)'].append(self.data[h].quantile(0.75))
            d['Стандартное отклонение'].append(self.data[h].std())
        return pd.DataFrame(d, columns)

    def remove_skipped(self, tab):
        removed = []
        categorical_index = []
        cont_index = []
        for i in tab.index:
            if tab['% пропусков'][i] > 60 and i != 'G_total':
                print(i, ' больше 60% пропусков')
                removed.append(i)
                continue
            if tab['Мощность'][i] == 1:
                print(i, ' мощность 1')
                removed.append(i)
                continue
            if tab['Мощность'][i] < 25:
                categorical_index.append(i)
            else:
                cont_index.append(i)
        self.data.drop(removed, axis=1, inplace=True)
        return cont_index, categorical_index

    def categorical_dist_properties(self, cat_index):
        d = {
            'Кол-во': [],
            '% пропусков': [],
            'Мощность': []
        }

        for j in (0, 1):
            d['Мода' + str(j + 1)] = []
            d['Частота моды' + str(j + 1)] = []
            d['% моды' + str(j + 1)] = []

        for i in cat_index:
            d['Кол-во'].append(self.data[i].count())
            d['% пропусков'].append(self.data[i].isna().sum() / len(self.data) * 100)
            d['Мощность'].append(self.data[i].nunique())
            vc = self.data[i].value_counts()
            for j in (0, 1):
                m = vc.index[j]
                m_count = vc[m]
                m_p = m_count / d['Кол-во'][cat_index.index(i)] * 100
                d['Мода' + str(j + 1)].append(m)
                d['Частота моды' + str(j + 1)].append(m_count)
                d['% моды' + str(j + 1)].append(m_p)
        return pd.DataFrame(d, cat_index)

    def get_gain_ratio(self, data):
        N = data.shape[0]
        n = int(np.log2(N)) + 1
        ct = pd.DataFrame(index=data.index, columns=data.columns)
        for column in ct:
            min = data[column].min()
            max = data[column].max()
            step = (max - min) / n
            for i in range(N):
                if not np.isnan(data[column][i]):
                    interval = int((data[column][i] - min) / step)
                    if interval == n:
                        interval -= 1
                    ct[column][i] = interval
                else:
                    ct[column][i] = -1
        #print(ct)
        ct.astype('int32')
        freq_T = np.zeros((n + 1, n), dtype=int)
        for i in range(N):
            freq_T[ct['G_total'][i] + 1, ct['КГФ'][i]] += 1
        #print(freq_T)

        info_T = 0
        for i in range(n + 1):
            for j in range(n):
                ft = freq_T[i, j]
                if ft != 0:
                    info_T -= (ft / N) * np.log2(ft / N)
        #print(info_T)
        gain_ratio = {}
        for column in ct.columns:
            if column != 'КГФ' and column != 'G_total':
                info_x_T = 0  # Оценка количеств информации после разбиения множества T по column
                split_info_x = 0
                for i in range(n):  # проходимся по классам разбиения
                    Ni = 0
                    freq_x_T = np.zeros_like(
                        freq_T
                    )  # Для каждого класса разбиения из column - мощности классов Cj
                    for j in range(N):
                        x = ct[column][j]
                        if x == i:
                            Ni += 1
                            freq_x_T[ct['G_total'][j] + 1, ct['КГФ'][j]] += 1
                    info_Ti = 0  # Оценка кол-ва информации для определения класса из Ti
                    if Ni != 0:
                        for i in range(n + 1):
                            for j in range(n):
                                if freq_x_T[i, j] != 0:
                                    info_Ti -= (freq_x_T[i, j] / Ni) * np.log2(freq_x_T[i, j] / Ni)
                        info_x_T += (Ni / N) * info_Ti
                        split_info_x -= (Ni / N) * np.log2((Ni / N))
                gain_ratio[column] = (info_T - info_x_T) / split_info_x

        vals = list(gain_ratio.values())
        length = len(vals)
        keys = list(gain_ratio.keys())
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.barh(keys, vals, align='center', color='green')
        for i in range(length):
            plt.annotate("%.2f" % vals[i], xy=(vals[i], keys[i]), va='center')
        plt.show()
        return gain_ratio

    def remove_emissions(self, cont_index, tab):
        normal_dist = ['Руст', 'Рзаб', 'Рлин', 'Рлин_2', 'Дебит кон нестабильный']
        for i in cont_index:
            if i in normal_dist:
                bot = tab['Среднее'][i] - 2 * tab['Стандартное отклонение'][i]
                top = tab['Среднее'][i] + 2 * tab['Стандартное отклонение'][i]
            else:
                x025 = tab['Первый квартиль(0.25)'][i]
                x075 = tab['Третий квартиль(0.75)'][i]
                bot = x025 - 1.5 * (x075 - x025)
                top = x075 + 1.5 * (x075 - x025)
            # print(i, bot, top)
            for j, row in self.data.iterrows():
                if self.data[i][j] < bot or self.data[i][j] > top:
                    # print(j)
                    if i == 'КГФ':
                        self.data.drop(index=j, inplace=True)
                    else:
                        self.data[i][j] = float('nan')
                        # if tab['% пропусков'][i] < 30:
                           # self.data[i][j] = tab['Медиана'][i]

        return self.data.reset_index(drop=True)

    def correlation_matrix(self):
        corr_matrix = self.data.corr().to_numpy()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(corr_matrix)
        ax.xaxis.set(ticks=np.arange(len(self.data.columns)), ticklabels=self.data.columns)
        ax.yaxis.set(ticks=np.arange(len(self.data.columns)), ticklabels=self.data.columns)
        ax.xaxis.set_tick_params(rotation=90)
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, '{:.2f}'.format(corr_matrix[i, j]), ha='center', va='center', color='w')
        plt.show()
        return corr_matrix

    def drop(self, corr_matrix, gain_ratio):
        dropped = []
        for i in range(len(self.data.columns)):
            col1 = self.data.columns[i]
            if col1 != 'КГФ' and col1 != 'G_total':
                for j in range(i):
                    col2 = self.data.columns[j]
                    if col2 in dropped:
                        continue
                    if col2 != 'КГФ' and col2 != 'G_total':
                        if corr_matrix[i, j] > 0.9:
                            drop_f = True
                            for k in range(len(self.data.columns)):
                                col3 = self.data.columns[k]
                                dif = abs(corr_matrix[i, k] - corr_matrix[j, k])
                                if col3 in dropped:
                                    continue
                                if dif > 0.25:
                                    drop_f = False
                            if drop_f:
                                print(f'{corr_matrix[i, j]} {col1} {gain_ratio[col1]}  -  {col2} {gain_ratio[col2]}')
                                if gain_ratio[col1] > gain_ratio[col2]:
                                    dropped.append(col2)
                                else:
                                    dropped.append(col1)
        print('\ndropped', dropped)
        self.data.drop(columns=dropped, inplace=True)
        self.data.to_csv('output.csv', sep=';', index=False, encoding='windows-1251')
