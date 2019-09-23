# -*- coding:utf-8 -*-
import torch
import jieba
import torch.utils.data as data


PATH = "F:/RNN/Embedding/douzhankuangchao.txt"
LABEL = "F:/RNN/Embedding/lexicon.txt"
# with open(LABEL, "w") as f:
#     lexicon = []
#     lines = open(PATH, "r").readlines()
#     for line in lines:
#         line = line.replace("，", "").replace("。", "").replace("：", "").replace("？", "").replace("“", "")\
#                     .replace("”", "").replace("《", "").replace("》", "").replace(".", "").replace("、", "")\
#                     .replace("…", "").replace("$", "").replace("；", "").replace("！", "").replace("~", "").strip()
#         lexicon.extend(list(jieba.cut(line)))
#     database = list(set(lexicon))
#     f.write("{}\n{}".format(lexicon, database))


class MyData(data.Dataset):
    def __init__(self, path):
        super(MyData, self).__init__()
        with open(path) as f:
            self.lexicon, self.database = f.readlines()
        self.lexicon, self.database = eval(self.lexicon), eval(self.database)  # 将字符串列表转化为列表

    def __len__(self):
        return len(self.database)-4  # 防止index+2溢出

    def __getitem__(self, idx):
        idx = idx+2  # 初始索引从2开始
        sentence, target = [], []
        for i in range(-2, 3):  # 前后共5个字符
            index = idx+i
            sentence.append(self.lexicon[index])
            target.append(self.database.index(self.lexicon[index]))
        target = torch.tensor(target).long()
        return sentence, target


mydata = MyData(LABEL)
# print(mydata[0])

