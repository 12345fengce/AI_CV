# -*- coding:utf-8 -*-
import os
from pydub import AudioSegment


PATH = "F:/StyleGAN/origin/"
SAVE = "F:/StyleGAN/"

def handle(path: str, save: str):
    for i in range(11, 21):
        file = path+"{}.aac".format(i)
        aac = AudioSegment.from_file(file, format="aac")  # 打开文件
        *ls, last = aac[::1000]  # 每秒区分  单位毫秒
        for j, audio in enumerate(ls):
            dir = save+"0{}".format(i)
            if not os.path.exists(dir):
                os.makedirs(dir)
            audio.export(dir+"/{}.wav".format(j), format="wav")  # 保存wav文件

if __name__ == '__main__':
    handle(PATH, SAVE)
