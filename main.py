from PIL import Image
import numpy as np
import os
import struct

from threading import Thread
import subprocess
import time
import logging

import threading

lock = threading.Lock()



IS_PART_DATASET = False


class LoggingHelper(object):
    INSTANCE = None

    def __init__(self):
        if LoggingHelper.INSTANCE is not None:
            raise ValueError("An instantiation already exists!")

        # os.makedirs(Temp_Path, exist_ok=True)
        self.logger = logging.getLogger()

        logging.basicConfig(filename='LOGGER', level=logging.INFO)

        fileHandler = logging.FileHandler('./msg.log')
        streamHandler = logging.StreamHandler()

        fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler.setFormatter(fomatter)
        streamHandler.setFormatter(fomatter)

        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)

    @classmethod
    def get_instace(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = LoggingHelper()
        return cls.INSTANCE

    @staticmethod
    def diff_time_logger(messege, start_time):
        LoggingHelper.get_instace().logger.info("[{}] :: running time {}".format(messege, time.time() - start_time))




logger = LoggingHelper.get_instace().logger

yuvpath = './DIV2K_train_YUV'

class PNGtoYUV(object):
    os.makedirs(yuvpath, exist_ok=True)
    def __init__(self, start, end):
        self.filelist = self.getFileList('./DIV2K_train_HR', start, end)
        self.saveYUV()
    def saveYUV(self):
        for file in self.filelist:
            im = Image.open(file)
            size = im.size
            # yuvim = im.convert('YCbCr')
            rgbarr = np.asarray(im)
            width = size[0]
            height = size[1]
            if width%8 != 0:
                width -= width%8
            if height%8 != 0:
                height -= height%8
            rgbarr = rgbarr[:height, :width, :]

            yuvarr = self.RGB2YUV(rgbarr)

            #
            # with open(os.path.join(self.savepath,'444test.yuv'), 'wb') as f:
            #     pixnum = str(size[0]*size[1]*3)
            #     f.write(struct.pack(pixnum + 'B', *(yuvarr.astype('uint8').transpose((2,0,1)).flatten())))
            u=  self.downSamplingUV(yuvarr[:,:,1])
            v = self.downSamplingUV(yuvarr[:,:,2])
            flattenYUV = np.concatenate((yuvarr[:,:,0].flatten(), u.flatten(), v.flatten()), axis =0).astype('uint8')

            newName = str(os.path.basename(file).split('.png')[0])+'_' + str(width) + 'x' + str(height) + '_8bit_P420.yuv'
            newPath = os.path.join(yuvpath, newName)
            with open(newPath, 'wb') as f:
                pixnum = str(width*height + (width*height)//2)
                f.write(struct.pack(pixnum + 'B', *flattenYUV))

    def RGB2YUV(self, rgb):

        m = np.array([[0.29900, -0.16874, 0.50000],
                      [0.58700, -0.33126, -0.41869],
                      [0.11400, 0.50000, -0.08131]])

        yuv = np.dot(rgb, m)
        yuv[:, :, 1:] += 128.5
        return yuv.astype('uint16')

    def downSamplingUV(self, uv):
        pos00 = uv[::2, ::2]
        pos10 = uv[1::2, ::2]
        ruv = (pos00 + pos10 + 1) >> 1
        return ruv.astype('uint8')



    def getFileList(self, dir, start, end, pattern='.png'):
        matches = []
        # for root, dirnames, filenames in os.walk(dir):
        #     for filename in filenames:
        for filename in os.listdir(dir):
            if filename.endswith(pattern):
                if IS_PART_DATASET:
                    if (int)(str(filename.split('.')[0]))>start and (int)(str(filename.split('.')[0]))<=end:
                        matches.append(os.path.join(dir, filename))
                else:
                    matches.append(os.path.join(dir, filename))
        return matches



class Encode(object):
    tmp_path = './temp'
    log_path = './log'
    bin_path = './bin'
    encoder_path = './EncoderApp.exe'
    cnd_path = './encoder_intra_vtm.cfg'

    def __init__(self):
        self.qplist = ['22', '27', '32', '37']
        self.cpu_num = os.cpu_count()
        if self.cpu_num>32:
            self.cpu_num = 32
        self.cpu_num -= 4
        self.running_cpu = 0
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.bin_path, exist_ok=True)


    def get_enc_command(self, name, qp):
        fullname = 'AI_PNG_'+name + '_Q' + qp
        binpath = os.path.join(self.bin_path, fullname + '.bin')
        seqpath = os.path.join(self.tmp_path, name + '.cfg')
        logpath = os.path.join(self.log_path, fullname + '.log')
        enc_command = self.encoder_path + ' -c ' + self.cnd_path + ' -c ' + seqpath + ' -q ' + qp + ' -b ' + binpath
        return enc_command, logpath

    def make_seqcfg_file(self, img_path):
        base_img_path  =  (str)(os.path.basename(img_path).split('.')[0])
        wxh = base_img_path.split('_')[1]
        width = (int)(wxh.split('x')[0])
        height = (int)(wxh.split('x')[1])
        filepath = os.path.join(self.tmp_path, base_img_path + '.cfg')
        with open(filepath, 'w') as f:
            f.write("InputFile : %s\n" % (img_path))
            f.write("InputBitDepth : %s\n" % ('8'))
            f.write("InputChromaFormat : %s\n" % ('420'))
            f.write("FrameRate : %s\n" % ('30'))
            f.write("FrameSkip : %s\n" % ('0'))
            f.write("SourceWidth : %s\n" % (width))
            f.write("SourceHeight : %s\n" % (height))
            f.write("FramesToBeEncoded : %s\n" % ('1'))
            f.write("Level : %s\n" % ('5.1'))
        return filepath

    def getFileList(self, dir, pattern='.yuv'):
        matches = []
        # for root, dirnames, filenames in os.walk(dir):
        #     for filename in filenames:
        for filename in os.listdir(dir):
            if filename.endswith(pattern):
                matches.append(os.path.join(dir, filename))
        return matches


    def runProcess(self,  imgpath, qp):
        cndpath = self.make_seqcfg_file(imgpath)
        name = (str)(os.path.basename(imgpath).split('.')[0])
        command, logpath = self.get_enc_command(name=name, qp = qp)
        logger.info('Start Encode : %s' %(name + '_Q' + qp))
        with open(logpath, 'w') as fp:
            sub_proc = subprocess.Popen(command, stdout = fp)
            sub_proc.wait()
        os.remove(cndpath)
        lock.acquire()
        self.running_cpu -=1
        lock.release()
        return

    def runThread(self):
        yuvlist = self.getFileList(yuvpath)
        for qp in self.qplist:
            for imgpath in yuvlist:
                while self.running_cpu >= self.cpu_num:
                    pass
                t = Thread(target=self.runProcess, args = (imgpath, qp, ))
                t.daemon = True
                t.start()
                lock.acquire()
                self.running_cpu += 1
                lock.release()




if __name__=='__main__':
    savepng = PNGtoYUV(600, 800)
    enc = Encode()
    enc.runThread()
