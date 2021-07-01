from models import exif
from PIL import Image
from PIL.ExifTags import TAGS
import os
import cv2
import random
from lib.utils import benchmark_utils, util,io
import numpy as np
wrong_tags =[545,546,547,548,544,549]
right_tags = ["ISO2","StopsAboveBaseISO","ExposureCompensation","BrightnessValue","FirmwareVersion","Info"]
# path to the image or video


def extract_exif():
    path = r"/content/drive/MyDrive/foto/foto/"
    dir = os.listdir(path)

    right_dir = []
    for i in dir:
        if "D" in i and "_" in i:
            s1 = os.path.join(path,i)
            s1 = os.path.join(s1,"orig")
            right_dir.append(s1)
    dict = {}
    image_list = []
    index = 0
    for dir in right_dir:
        directory = os.listdir(dir)
        for elem in directory:
            new_dir = os.path.join(dir,elem)
            
            # read the image data using PIL
            image = Image.open(new_dir)
            
            if new_dir not in image_list : image_list.append(new_dir)
            # extract EXIF data
            exifdata = image.getexif()
            # iterating over all EXIF data fields
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                #if isinstance(data, bytes):
                #    data = data.decode('utf-8')
                #data = str(data).strip(" ")
                if tag not in dict.keys():
                    dict[tag] = [[data,[elem]]]
                else:
                    flag = False
                    for i in range(len(dict[tag])):
                        if dict[tag][i][0] == data:
                            dict[tag][i][1].append(elem)
                            flag = True
                    if flag == False:
                        dict[tag].append([data,[elem]])

            index+=1
    for i in range(len(wrong_tags)):
        x = dict.pop(wrong_tags[i])
        dict[right_tags[i]] = x

    print("[INFO] Extracted dict")
    return dict,image_list

def random_list(list):
    second_list = []
    for i in range(len(list)):
        if i % 300 == 0:
            second_list.append(list[i])
        else:
            second_list.append(random.choice(list))
    print("[INFO] Generated second list")
    return second_list

def generate_label(first,second):
    exif_lbl = []
    if(len(first)!=len(second)):
        print("list len do not match")
        return
    else:
        for i in range(len(first)):
            im1 = Image.open(first[i])
            im2 = Image.open(second[i])
            exif1 = im1.getexif()
            exif2 = im2.getexif()
            list_tag1={}
            list_tag2={}
            for tag_id in exif1:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exif1.get(tag_id)
                #if isinstance(data, bytes):
                #    data = data.decode()
                #data = str(data).strip(" ")
                list_tag1[tag] = data
            
            for tag_id in exif2:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exif1.get(tag_id)
                #if isinstance(data, bytes):
                #    data = data.decode()
                #data = str(data).strip(" ")
                list_tag2[tag] = data
            shared_tags = []
            
            for elem in list_tag1.keys():
                if elem in list_tag2.keys():
                    if(list_tag1[elem] == list_tag2[elem]):
                        shared_tags.append(1)
                    else:
                        shared_tags.append(0)
                else:
                    shared_tags.append(0)
            
            for elem in list_tag2.keys():
                if elem not in list_tag1.keys():
                    shared_tags.append(0)
            
            exif_lbl.append(shared_tags)
        print("[INFO] Label extracted")
        return exif_lbl
    
def cropping_list(first,second):
    tmp1 = []
    tmp2 = []
    
    for i in range(len(first)):
        print(i)
        x = cv2.imread(first[i])[:,:,[2,1,0]]
        y = cv2.imread(second[i])[:,:,[2,1,0]]
        im1 = util.random_crop(x,[128,128])
        im2 = util.random_crop(y,[128,128])
        tmp1.append(im1)
        tmp2.append(im2)
    
    print("[INFO] Images cropped")
    return tmp1,tmp2

"""
####################################################ORIGINAL #################################################
# path to the image or video
# path to the image or video
path = os.getcwd()
path+=r"\foto"
directory = os.listdir(path)
# read the image data using PIL
for elem in directory:
    new_dir = os.path.join(path,elem)
    image = Image.open(new_dir)
    # extract EXIF data
    exifdata = image.getexif()
    # iterating over all EXIF data fields
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes 
        if isinstance(data, bytes):
            data = data.decode()
        data = str(data).strip(" ")
        print(f"{tag:25}: {data}")
    print()
    print()"""