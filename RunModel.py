from pandas import read_csv
from keras.regularizers import L1L2
from tools.utils import model_configs, evaluate_model

if __name__ == '__main__':

    reg = L1L2(l1=0.01, l2=0.01)
    dataset = read_csv('input/processed_data_for_input.csv',
                       header=0,
                       index_col=0)
    cfg_list = model_configs()
    numoflayers = 3
    out_file_csv = "output/" + str(numoflayers) + 'layers grids.csv'
    is_first = 0
    for cfg in cfg_list:
        try:
            evaluate_model(
                dataset.values, is_first,
                out_file_csv,
                cfg)
            is_first = 1
            print('done')
        except:
            pass
