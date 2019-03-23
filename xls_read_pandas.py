import csv
import xlrd
import numpy as np
import pandas as pd


def csv_from_excel():
    wb = xlrd.open_workbook('温度.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    with open('温度.csv', 'w', newline='') as your_csv_file:
        # your_csv_file = open('温度.csv', 'w', newline='')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))


filename = '温度.xlsx'
df = pd.DataFrame(pd.read_excel(filename, header=[0, 1, 2], index_col=[0]))
values = df.values
Y = pd.DataFrame(values)
Y.dropna(axis=1, how='any', inplace=False)
# b = [i for i in range(Y.shape[1]) if (Y[- 1, i] == '' or Y[0, i] == '')]
# Y = np.delete(Y, b, axis=1)
