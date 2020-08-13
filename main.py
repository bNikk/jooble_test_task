import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import sys


def main(file_name, scaler):

    if scaler == 'zscore':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    elif scaler == 'maxabs':
        sc = MaxAbsScaler()

    df = pd.read_csv(file_name, sep='\t', header=0)
    temp = df['features'].str.split(',', expand=True)

    col_name = []
    for i in range(1, len(temp.columns)):
        (col_name.append(f'feature_{temp[0][0]}_{i}'))

    temp = temp.astype(int)
    temp = sc.fit_transform(temp.iloc[:, 1:])
    temp = pd.DataFrame(temp)
    temp = temp.set_axis(col_name, axis=1)

    temp['max_feature_2_abs_mean_diff '] = (temp.max(axis=1) - temp.max(axis=1).mean()).astype(float)
    temp['max_feature_2_index'] = temp.idxmax(axis="columns").str[10:].astype(int)
    temp = pd.concat([df.iloc[:, 0], temp], axis=1)

    return temp.to_csv(f'{file_name[:-4]}_proc.tsv', sep='\t', index=False)


if __name__ == '__main__':

    main(sys.argv[1], sys.argv[2])

