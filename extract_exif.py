from PIL import Image
from PIL.ExifTags import TAGS
import os
wrong_tags =[545,546,547,548,544,549]
right_tags = ["ISO2","StopsAboveBaseISO","ExposureCompensation","BrightnessValue","FirmwareVersion","Info"]
# path to the image or video
def extract_exif():
    path = os.getcwd()
    path = os.path.join(path,"foto")
    dir = os.listdir(path)

    right_dir = []
    for i in dir:
        if "D" in i and "_" in i:
            s1 = os.path.join(path,i)
            s1 = os.path.join(s1,"orig")
            right_dir.append(s1)
    dict = {}
    index = 0
    for dir in right_dir:
        directory = os.listdir(dir)
        for elem in directory:
            new_dir = os.path.join(dir,elem)
            
            # read the image data using PIL
            image = Image.open(new_dir)

            # extract EXIF data
            exifdata = image.getexif()
            
            # iterating over all EXIF data fields
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                if isinstance(data, bytes):
                    data = data.decode()
                data = str(data).strip(" ")
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
    return dict

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