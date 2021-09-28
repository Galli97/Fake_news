from numpy.lib.function_base import append
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

##my path r"C:\Users\Adri\Desktop\VISIOPE\prova\foto"
## drive path r"/content/drive/MyDrive/foto/foto/"
def extract_exif():
    path = r"/content/drive/MyDrive/foto/foto/"
    dir = os.listdir(path)
    no_dir = open(r"/content/drive/MyDrive/foto/foto/chiavi.txt","r").read().splitlines()
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
            if exifdata is None:
                print("Sorry, image has no exif data.")
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

    #convert tags numbers to string
    for i in range(len(wrong_tags)):
        x = dict.pop(wrong_tags[i])
        dict[right_tags[i]] = x

    #remove tags with less elements than 30
    for key in list(dict):
        if str(key) in no_dir:
            dict.pop(key)
        else:
            i = len(dict[key])-1
            while(i>=0):
                if(len(dict[key][i][1])<30):
                    dict[key].pop(i)
                i=i-1
            
    print(len(dict.keys()))
    print("[INFO] Extracted dict")
    return dict,image_list,list(dict.keys())

def random_list(list):
    second_list = []
    for i in range(len(list)):
        if i % 300 == 0:
            second_list.append(list[i])
        else:
            second_list.append(random.choice(list))
    print("[INFO] Generated second list")
    return second_list

def generate_label(keys,first,second):
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
            shared_tags = []
            im1realkeys = []
            im1keys = []
            im2keys = []
            if str(first[i]) == str(second[i]):
                shared_tags = [1]*len(keys)
                print(f"stampati {len(keys)} Ones, array: {shared_tags}")
                print("[INDEX]: ",i)
            else:
                for elem in exif1:
                    tag = TAGS.get(elem, elem)
                    im1keys.append(tag)
                    im1realkeys.append(elem)
                
                for elem in exif2:
                    tag = TAGS.get(elem, elem)
                    im2keys.append(tag)
                

                for tag_id in keys:
                    if tag_id in im1keys and tag_id in im2keys:
                        #tag = TAGS.get(tag_id, tag_id)
                        exif_real = im1keys.index(tag_id)
                        exif_real = im1realkeys[exif_real]

                        data1 = exif1.get(exif_real)
                        data2 = exif2.get(exif_real)
                        if(str(data1) == str(data2)):
                            shared_tags.append(1)
                        else:
                            shared_tags.append(0)
                    else:
                        shared_tags.append(0)

            exif_lbl.append(shared_tags)
        print("[INFO] Label extracted")
        
        return exif_lbl


def save_np_arrays(tmp1,tmp2):
    with open('cropped_arrays.npy','wb') as f:
        np.save(f,tmp1)
        np.save(f,tmp2)

def get_np_arrays(file):
    with open(file,'rb') as f:
        tmp1 = np.load(f)
        tmp2 = np.load(f)
    return tmp1,tmp2

def cropping_list(first,second):
    N = len(first)
    tmp1 = np.empty((N, 128, 128, 3), dtype=np.uint8)
    tmp2 = np.empty((N, 128, 128, 3), dtype=np.uint8)
    for i in range(N):
        if (i % 77 == 0): print(i) #77
        x = cv2.imread(first[i])[:,:,[2,1,0]]
        y = cv2.imread(second[i])[:,:,[2,1,0]]

        patch1 = util.random_crop(x,[128,128])
        patch2 = util.random_crop(y,[128,128])
        tmp1[i] = patch1
        tmp2[i] = patch2
    
    print("[INFO] Images cropped")
    save_np_arrays(tmp1,tmp2)
    #a,b = get_np_arrays('cropped_arrays.npy')
    #if(np.array_equal(a,tmp1) and np.array_equal(b,tmp2)):
        #print("Corretti")
            

    return tmp1 ,tmp2