# -*- coding:utf-8 -*-
import os
import torch.nn.init as init
from pydub import AudioSegment


def handle(path: str, save: str):
    """tansform *.acc to *.wav
        time_len = 1s"""
    for i in range(11, 21):
        file = path+"{}.aac".format(i)
        aac = AudioSegment.from_file(file, format="aac")  # 打开文件
        *ls, last = aac[::1000]  # 每秒区分  单位毫秒
        for j, audio in enumerate(ls):
            dir = save+"0{}".format(i)
            if not os.path.exists(dir):
                os.makedirs(dir)
            audio.export(dir+"/{}.wav".format(j), format="wav")  # 保存wav文件


def normalize(path: str, sample_rate=44100, channels=2):
    """tansform *wav to *.wav
        sample_rate = 44100
        chnnels=2"""
    for dir in os.listdir(path):
        dir = path+"/"+dir
        for file in os.listdir(dir):
            file = dir+"/"+file
            vedio = AudioSegment.from_wav(file)
            stereo = vedio.set_frame_rate(sample_rate).set_channels(channels)
            stereo.export(file, format='wav')


def cut(path: str):
    """transform *.wav to *.wav
        length = 1000"""
    for dir in os.listdir(path):
        dir = path+"/"+dir
        for file in os.listdir(dir):
            file = dir+"/"+file
            vedio = AudioSegment.from_wav(file)
            if len(vedio) < 1000:
                vedio = vedio+vedio[len(vedio)-1000:]
            elif len(vedio) < 1500:
                vedio = vedio[:1000]
            else:
                vedio = vedio[500:1500]
            vedio.export(file, format='wav')


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ["Conv2d", "Linear"]:
        init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
    elif classname.find("PReLU") != -1:
        init.constant_(m.weight.data, 0.01)


