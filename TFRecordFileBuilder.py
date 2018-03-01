import re, os, sys, glob, xlrd, shutil, mimetypes, random, argparse, math
from os import sep
import os.path as path
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

class BUtils(object):
    @staticmethod  # 去掉目录路径结尾的路径分隔符
    def remove_end_sep(dpath):
        """去掉目录路径结尾的分隔符"""
        if os.path.isfile(dpath): BUtils.prompt(-1, "Not a tree: {}".format(dpath))
        if not dpath.endswith(sep):
            return dpath
        else:
            return BUtils.remove_end_sep(dpath[:-1])

    @staticmethod
    def process_bar(i, n, pref_str='', suff_str='', char='=', num_chars=100):
        '''
        :param i: 计数器
        :param n: 计数器最大值
        :param char: 进度条符号
        :param pref_str: 进度条前置字符串
        :param suff_str: 进度条后置字符串
        '''
        i += 1
        num = i * num_chars // n
        pre = char * num
        pro = '>' + ' ' * (num_chars - 1 - num) if num - num_chars else ''
        numerator = ' ' * (len(str(n)) - len(str(i))) + str(i)
        if num - num_chars:
            sys.stdout.write("%s|%s/%d|%s%s|%s%%|%s\r" % (
            pref_str, numerator, n, pre, pro, ' ' + str(num * 100 // num_chars), suff_str))
        else:
            sys.stdout.write("%s|%s/%d|%s%s|%s%%|%s\n" % (pref_str, numerator, n, pre, pro, 100, suff_str))

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False


class TFRecordFileBuilder(object):

    def __init__(self, img_dir, std_shape, train_ratio):
        self.img_dir = BUtils.remove_end_sep(img_dir) # 原始图像位于一个目录下面（无子目录）
        self.std_shape = std_shape # 网络标准输入尺寸
        # generate self.data_dict: file_name -> age
        self.load_xlsxs(0, 2)
        # generate self.train_set and self.eval_set
        self.split_ET(train_ratio)

        self.train_tfr = "{}{}train.tfr".format(path.dirname(self.img_dir), sep)
        self.eval_tfr = "{}{}eval.tfr".format(path.dirname(self.img_dir), sep)
        self.make_train_tfr()
        self.make_eval_tfr()

        with open("{}{}info.txt".format(path.dirname(self.img_dir), sep), 'w') as writer:
            t, e = len(self.train_set), len(self.eval_set)
            string = "num_train_samples\t{}\n".format(t)
            string += "num_eval_samples\t{}\n".format(e)
            string += "num_samples\t{}".format(t+e)
            writer.write(string)

    def load_xlsxs(self, keyIdx, valueIdx, keyTypeFn=str, valueTypeFn=int):
        '''加载某个目录下的所有xlsx文件到一个字典'''
        def load_xlsx(filePath, keyIdx, valueIdx, keyTypeFn, valueTypeFn):
            """
            加载标签文件,建立索引-年龄字典
            :param filePath: string -> xlsx文件路径
            :param keyIdx: int -> 特征列索引, 0,1,...
            :param valueIdx: int -> 标签列索引
            :param keyTypeFn: fn -> 特征列数据类型
            :param valueTypeFn: fn -> 标签列数据类型
            :return: {} -> 特征列-标签列字典
            """
            labels = {}
            for sheet in xlrd.open_workbook(filename=filePath).sheets():
                for sampleIdx in range(sheet.nrows):
                    splitSample = sheet.row_values(sampleIdx)
                    key, value = splitSample[keyIdx], splitSample[valueIdx]
                    if not value or not BUtils.is_number(value): continue
                    if labels.get(key, 'NOT_EXISTENT') != 'NOT_EXISTENT':
                        print("[WARNING] Repeated key found: '{}', overwrite it.".format(key))
                    labels[keyTypeFn(key)] = valueTypeFn(value)
            return labels
        d = {}
        rootDir = self.img_dir
        for file in glob.glob(rootDir + sep + "*.xlsx"):
            d.update(load_xlsx(file, keyIdx, valueIdx, keyTypeFn, valueTypeFn))
        # test if file exist
        dd = {}
        for f in d:
            full_name = rootDir + sep + f + '.jpg'
            if path.exists(full_name):
                dd[f] = d[f]
        self.data_dict = dd

    def split_ET(self, train_ratio, exclude_kw='zla'):
        '''按照self.data_dict将图片分成训练集和测试集'''
        fs = [f for f in self.data_dict.keys() if exclude_kw not in f] # 不包含zla的文件
        random.shuffle(fs)
        fs_len = len(fs)
        num_t = int(fs_len * train_ratio)
        self.train_set = fs[:num_t]
        self.eval_set = fs[num_t:]

    def make_train_tfr(self):
        with tf.python_io.TFRecordWriter(self.train_tfr) as writer:
            num_t = len(self.train_set)
            for idx in range(num_t):
                BUtils.process_bar(idx, num_t, "[INFO] Make train_tfr ", '', '=', 25)
                file_name = self.train_set[idx]
                label = self.data_dict.get(file_name, "NOT_EXIST")
                if label == "NOT_EXIST":
                    print("[INFO] train: abel of {} not exist in data_dict".format(file_name))
                    continue
                img = Image.open(self.img_dir+sep+file_name+'.jpg').convert('L')
                ## 剪切以统一尺寸
                crop_box = self.get_crop_box(img)
                img = img.crop(crop_box)
                # 滑动增强的同时保存到TFRecord
                self.slid_and_tfr(writer, img, label, kernel=(2420, 1260), stride=(80, 40), resize=self.std_shape)

    def make_eval_tfr(self):
        with tf.python_io.TFRecordWriter(self.eval_tfr) as writer:
            num_e = len(self.eval_set)
            for idx in range(num_e):
                BUtils.process_bar(idx, num_e, "[INFO] Make eval_tfr ", '', '=', 25)
                file_name = self.eval_set[idx]
                label = self.data_dict.get(file_name, "NOT_EXIST")
                if label == "NOT_EXIST":
                    print("[INFO] eval: label of {} not exist in data_dict".format(file_name))
                    continue
                img = Image.open(self.img_dir+sep+file_name+'.jpg').convert('L')
                ## 剪切以统一尺寸
                crop_box = self.get_crop_box(img, shape=(1260, 2420))
                img = img.crop(crop_box)
                ## 缩放保存
                img = img.resize(size=self.std_shape)
                self._write_to_tfrecord(writer, img, label)

    def get_crop_box(self, img, shape=(1300,2500)):
        """返回图片居中的大小为shape的box供crop使用
        Args:   img: 灰度图（2D）
        """
        img_arr = np.array(img)
        imgH, imgW = img_arr.shape
        cropTop = (imgH - shape[0]) // 2
        cropLeft = (imgW - shape[1]) // 2
        return (cropLeft, cropTop, cropLeft+shape[1], cropTop+shape[0])

    def slid_and_tfr(self, writer, img, label, kernel, stride=(1, 1), resize=None):
        """给定原始图片进行滑动并且保存为TFRecord"""
        fsize = img.size  # (2500,1300)
        for h in range(0, fsize[0] - kernel[0] + 1, stride[0]):
            for v in range(0, fsize[1] - kernel[1] + 1, stride[1]):
                sub_img = img.crop(box=(h, v, kernel[0] + h, kernel[1] + v))
                if resize != None: sub_img = sub_img.resize(size=resize)
                self._write_to_tfrecord(writer, sub_img, label)

    def _write_to_tfrecord(self, writer, img, label):
        img_raw = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())


def read_and_decode(filename, std_shape=(447, 447, 1)):
    '''根据TFRecord文件定义图片数据和标签的读取方式
    参数：
        filename：TFRecord文件路径
        std_shape：TF加载图片数据后有一个reshape的过程
    返回：tuple(img, label)
    '''
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw' : tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, std_shape)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

        
        
if __name__ == '__main__':
    ## 生成TFRecord
    # src_dir = "/home/chenyin/dataDir/dentale_data/raw-data/train"
    # cwd = os.getcwd()
    # D = DataLoader( src_dir, (447, 447) )
    # D.load_train_data("/home/chenyin/dataDir/dentale_data/raw-data/train-447.tfr")

    # src_dir = "/home/chenyin/dataDir/dentale_data/raw-data/eval"
    # cwd = os.getcwd()
    # D = DataLoader( src_dir, (447, 447) )
    # D.load_eval_data("/home/chenyin/dataDir/dentale_data/raw-data/eval-447.tfr")

    src_dir = "/home/chenyin/dataDir/dentale_data/raw-data/raw"
    D = TFRecordFileBuilder(src_dir, (447, 447), 0.9)
    # for k in D.data_dict: print("{} -> {}".format(k, D.data_dict[k]))