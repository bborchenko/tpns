from matplotlib import pyplot as plt

from data_parser import Parser
from logic import Logic


def main():
    ps = Parser('ID_data_mass_18122012.csv')

    data = ps.get_data()
    logic = Logic(data)

    tab_dist = logic.get_properties()
    cont_index, cat_index = logic.remove_skipped(tab_dist)
    tab_cat = logic.categorical_dist_properties(cat_index)
    # print(tab_cat)
    data.rename(columns={cat_index[i]: cat_index[i] + '_категориальный' for i in range(len(cat_index))})
    data.hist(bins=50, figsize=(20, 20), color='green')
    plt.show()
    data = logic.remove_emissions(cont_index, tab_dist)
    data.hist(bins=50, figsize=(20, 20), color='green')
    plt.show()

    gain_ratio = logic.get_gain_ratio(data)
    corr_matrix = logic.correlation_matrix()
    logic.drop(corr_matrix, gain_ratio)


if __name__ == '__main__':
    main()
