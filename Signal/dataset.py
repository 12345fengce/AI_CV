# -*- coding:utf-8 -*-
import xlrd
import torch
import numpy as np
import torch.utils.data as data


PATH = "F:/SIGNAL/data1.xls"


class MyData(data.Dataset):
    def __init__(self, path):
        sheets = xlrd.open_workbook(filename=path, encoding_override="utf-8")
        sheet = sheets.sheet_by_index(0)
        self.database = np.array(sheet.col_values(8), dtype=np.int64)

    def __len__(self):
        return len(self.database)-9

    def __getitem__(self, idx):
        database = self.database/np.max(self.database)
        target = torch.Tensor(database[idx+9: idx+10])
        data = torch.Tensor(database[idx: idx+9])
        return data, target, np.max(self.database)


if __name__ == '__main__':
    mydata = MyData(PATH)
    print(mydata[0])