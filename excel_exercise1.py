import xlrd
import numpy as np


def xls_print_cell_value(excel_path):
    """
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    # print(table[0].get_rows())#returns a indicaator of messy code
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an indicator(virtual adress)指针 for iterating through each row
    for row in rows:
        for i in range(0, sheet1.ncols):  # iterating through each col of one particular row
            print(row[i].value, end=' ')  # print out the value of the one colum
        print('\n')  # at the last colum of the iterating row,print a line break and start with the next row


def xls_txt(excel_path, txt_flename):
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()
    sheet1 = table[0]
    the_output = np.array([])
    for i in range(0, sheet1.ncols):  # iterating through each col of one particular row
        rows = sheet1.get_rows()  # this return a indicator which points at the beginning of an row
        for row in rows:
            # read one row then add each item into one list
            temporary_array = np.array([row[i].value])
            the_output = np.append(the_output, temporary_array)
    the_output = np.reshape(the_output, (sheet1.ncols, -1)).transpose()
    np.savetxt(txt_flename, the_output)


def xls_print_cell_values(excel_path, row_start=0, col_start=0, row_end=-1, col_end=-1):
    """
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
            col_start:start of the colum
            col_end:end of the colum
            row_start:start of the row
            row_end:end of the row
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an generator for iterating through each row
    i = 0
    for row in rows:

        if i < row_start:
            i += 1  # in case that the cycle continued then the i didn't +1
            continue
        if row_end != -1:  # the row_end = -1 turns to cycle to the end of the row
            if i > row_end:
                break
        if row_end == -1:
            row_end = sheet1.ncols
        if col_start < col_end:  # in case that col_start=col_end then there won't exist range()
            for j in range(col_start, col_end):
                print(row[j].value, end=' ')
        else:
            print(row[col_start].value, end=' ')
        print('\n')
        i += 1  # at the end of one cycle ,the i+1 so that there will be one cycle as i=input


def xls_arrays(excel_path, row_start=0, col_start=0, row_end=-1, col_end=-1):
    """
    !!row_start starts from 0 ,and the same as col_start
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
            col_start:start of the colum
            col_end:end of the colum
            row_start:start of the row
            row_end:end of the row
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an generator for iterating through each row
    i = 0
    the_output = np.array([])
    for row in rows:

        if i < row_start:
            i += 1  # in case that the cycle continued then the i didn't +1
            continue
        if row_end != -1:  # the row_end = -1 turns to cycle to the end of the row
            if i > row_end:
                break
        if col_end == -1:
            col_end = sheet1.ncols - 1
        if col_start < col_end:  # in case that col_start=col_end then there won't exist range()
            for j in range(col_start, col_end + 1):
                temporary_array = np.array([row[j].value])
                the_output = np.append(the_output, temporary_array)
        else:
            temporary_array = np.array([row[col_start].value])
            the_output = np.append(the_output, temporary_array)
        i += 1  # at the end of one cycle ,the i+1 so that there will be one cycle as i=input
    if row_end != -1:
        the_output = np.reshape(the_output, (row_end - row_start + 1, -1))
    else:
        the_output = np.reshape(the_output, (i - row_start, -1))
    return the_output


if __name__ == '__main__':
    excel_path = '温度.xlsx'
    txt_flename = "result.txt"
    # xls_txt(excel_path, txt_flename)
    z = xls_arrays(excel_path, 3, 1, 2002, -1)

'''def xls_arrays(excel_path, row_start=0, col_start=0, row_end=-1, col_end=-1):
    """
    !!row_start starts from 0 ,and the same as col_start
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
            col_start:start of the colum
            col_end:end of the colum
            row_start:start of the row
            row_end:end of the row
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an generator for iterating through each row
    i = 0
    the_output = np.array([])
    for row in rows:

        if i < row_start:
            i += 1  # in case that the cycle continued then the i didn't +1
            continue
        if row_end != -1:  # the row_end = -1 turns to cycle to the end of the row
            if i > row_end:
                break
        if col_end == -1:
            col_end = sheet1.ncols - 1
        if col_start < col_end:  # in case that col_start=col_end then there won't exist range()
            for j in range(col_start, col_end + 1):
                temporary_array = np.array([row[j].value])
                the_output = np.append(the_output, temporary_array)
        else:
            the_output = np.vstack
        i += 1  # at the end of one cycle ,the i+1 so that there will be one cycle as i=input
    if row_end != -1:
        if the_output == np.array([]):
            the_output = sheet1.row_values(i)
        else:
            the_output = np.vstack((the_output, sheet1.row_values(i)))
    else:
        the_output = sheet1.row_values(i)
    return the_output'''
'''def xls_arrays(excel_path, row_start=0, col_start=0, row_end=-1, col_end=-1):
    """
    !!row_start starts from 0 ,and the same as col_start
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
            col_start:start of the colum
            col_end:end of the colum
            row_start:start of the row
            row_end:end of the row
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an generator for iterating through each row
    i = 0
    the_output = np.array([])
    for row in rows:

        if i < row_start:
            i += 1  # in case that the cycle continued then the i didn't +1
            continue
        if row_end != -1:  # the row_end = -1 turns to cycle to the end of the row
            if i > row_end:
                break
        if col_end == -1:
            col_end = sheet1.ncols - 1
        if col_start < col_end:  # in case that col_start=col_end then there won't exist range()
            for j in range(col_start, col_end + 1):
                temporary_array = np.array([row[j].value])
                the_output = np.append(the_output, temporary_array)
        else:
            temporary_array = np.array([row[col_start].value])
            the_output = np.append(the_output, temporary_array)
        i += 1  # at the end of one cycle ,the i+1 so that there will be one cycle as i=input
    if row_end != -1:
        the_output = np.reshape(the_output, (row_end - row_start + 1, -1))
    else:
        the_output = np.reshape(the_output, (i - row_start, -1))
    return the_output
'''

'''excel_path = '1.xlsx'
book = xlrd.open_workbook(excel_path)
table = book.sheets()
sheet1 = table[0]
the_output = np.array([])
for i in range(0, sheet1.ncols):  # iterating through each col of one particular row
    rows = sheet1.get_rows()  # this return a indicator which points at the beginning of an row
    for row in rows:
        print(row[i].value, end=' ')  # print out the value of the one col
        # read one row then add each item into one list
        # temporary_list=np.array()
        try:
            temporary_array = np.array([row[i].value])
            the_output = np.append(the_output, temporary_array)
        except:
            print('line37')

    print('\n')  # at the last colum of the iterating row,print a line break and start with the next row
the_output = np.reshape(the_output, (sheet1.ncols, -1)).transpose()
print(the_output)'''

'''def xls_print_cell_values(excel_path, col_start=0, row_start=0, col_end=-1, row_end=-1):
    """
    :param excel_path: thexlrd.open_workbook()method returns the nassssay value
            col_start:start of the colum
            col_end:end of the colum
            row_start:start of the row
            row_end:end of the row
    :return: print our all the cell value in one sheet
    """
    book = xlrd.open_workbook(excel_path)
    table = book.sheets()  # returns A list of all sheets in the book.
    # print(table[0].get_rows())#returns a indicaator of messy code
    sheet1 = table[0]  # get one sheet(the first of all) in the pre-mentioned book
    rows = sheet1.get_rows()  # the method sheeet.get_rows() returns an indicator(virtual adress)指针 for iterating through each row
    print('number of cols:', sheet1.ncols)
    if col_end == -1:  # 如果没有给出col的结束值,将最大的col值输入
        col_end = sheet1.ncols
    # if row_end == -1:#同理.没有row默认值,将最大row输入

    for row in rows[row_start, row_end]:
        for i in range(col_start, col_end):  # iterating through each col of one particular row
            print(row[i].value, end=' ')  # print out the value of the one colum
        print('\n')  # at the last colum of the iterating row,print a line break and start with the next row

'''
