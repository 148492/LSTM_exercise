import numpy as np
import xlrd
import excel_exercise1 as ee1





if __name__ == '__main__':
    excel_path = '温度.xlsx'
    txt_flename = "result.txt"
    # xls_txt(excel_path, txt_flename)
    # ee1.xls_print_cell_values(excel_path, 4, 2, 104, 2)
    tensor1 = xls_arrays(excel_path, 4, 2, 104, 2)
    print(tensor1)

'''
row_start = 0
row_end = 100
col_start = 0
col_end = 12
excel_path = '温度.xlsx'
book = xlrd.open_workbook(excel_path)
table = book.sheets()  # returns A list of all sheets in the book.
# print(table[0].get_rows())#returns a indicaator of messy code
sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an indicator(virtual adress)指针 for iterating through each row

i = row_start
for row in rows:
    if i <= row_end:
        i += 1
    else:
        break
    for j in range(col_start, col_end):
        print(row[j].value, end=' ')
    print('\n')
print(type(rows))
'''
# data=np.loadtxt('1.txt', str, unpack=True)
# np.savetxt("result.txt", the_output)
# data = np.loadtxt('1.txt')
# a, b = 1, 2
# print(a, b)
