# -*- coding:utf-8 -*-
import train

if __name__ == '__main__':
    log = "./Plog.txt"
    mytrain = train.MyTrain("P")
    mytrain.run(log)