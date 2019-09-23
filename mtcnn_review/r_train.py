# -*- coding:utf-8 -*-
import train

if __name__ == '__main__':
    log = "./Rlog.txt"
    mytrain = train.MyTrain("R")
    mytrain.run(log)