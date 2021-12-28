from retinaface import RetinaFace
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import numpy as np
import os

#f64 = open("64x64.csv", "w")
#f128 = open("128x128.csv", "w")
ftarget = open("target.csv", "a", encoding="utf-8")
ferror = open("error.txt", "a", encoding="utf-8")

def extract_img(img_path):
    global f64, f128
    
    print(img_path)
    
    img = cv2.imread(img_path)

    try:
        resp = RetinaFace.detect_faces(img_path, threshold = 0.5)
    #print(resp)
    except:
        print("\t{} file corrupted".format(img_path))
        ferror.write("\t{} file corrupted".format(img_path))
        return

    i = 0
    for key in resp:
        identity = resp[key]
        facial_area = identity["facial_area"]
        #cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
        #facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        #plt.imshow(facial_img[:, :, ::-1])
        print(facial_area)
        crop = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        o64 = cv2.resize(gray, (64,64))
        cv2.imwrite("{}_{}_{}.png".format(img_path,i,64), o64)
        #n64 = np.asarray(o64).reshape(1,64*64)
        #np.savetxt(f64, n64, "%3d", delimiter=",")
        o128 = cv2.resize(gray, (128,128))
        cv2.imwrite("{}_{}_{}.png".format(img_path,i,128), o128)
        #n128 = np.asarray(o128).reshape(1,128*128)
        #np.savetxt(f128, n128, "%3d", delimiter=",")
        ftarget.write(img_path+"\r\n")
        #d = np.array(o64)
        #print(d)
        i+=1
    #sys.exit()

def cv2_imread(path):
    arr = np.fromfile(path, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def cv2_imwrite(path, img):
    global ferror
    result, encoded_img = cv2.imencode(".PNG", img)
    if result:
        f = open(path, "w+b")
        encoded_img.tofile(f)
        f.close()
    else:
        ferror.write("cv2_imwrite error at {}\r\n".format(path))

def extract_img2(img_path, dest_path):
    global ftarget, ferror
    print(img_path)
    
    #img = cv2.imread(img_path)
    img = cv2_imread(img_path)
    if (img is None):
        print("\t{} file corrupted".format(img_path))
        ferror.write("\t{} file corrupted".format(img_path))
        return
    #print(img)
    #print(img.shape)

    try:
        #resp = RetinaFace.detect_faces(img_path, threshold = 0.5)
        resp = RetinaFace.detect_faces(img, threshold = 0.5)
    #print(resp)
    except Exception as e:
        print("\t{} file corrupted : Retina Failed".format(img_path))
        print(e)
        ferror.write("\t{} file corrupted: Retina Failed".format(img_path))
        return

    i = 0
    nFaces = len(resp)
    if nFaces != 1:
        print("\t{} faces.. skip".format(nFaces))
        return


    for key in resp:
        identity = resp[key]
        facial_area = identity["facial_area"]
        #cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
        #facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        #plt.imshow(facial_img[:, :, ::-1])
        print("\t",facial_area[3]-facial_area[1], facial_area[2]-facial_area[0])
        ftarget.write("{}|{}|{}\r\n".format(img_path, facial_area[3]-facial_area[1], facial_area[2]-facial_area[0]))
        crop = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        o64 = cv2.resize(gray, (64,64))
        outFile = "{}_{}.png".format(dest_path,64)
        print(outFile)
        #cv2.imwrite(outFile, o64)
        cv2_imwrite(outFile, o64)
        #n64 = np.asarray(o64).reshape(1,64*64)
        #np.savetxt(f64, n64, "%3d", delimiter=",")
        o128 = cv2.resize(gray, (128,128))
        outFile = "{}_{}.png".format(dest_path,128)
        print(outFile)
        #cv2.imwrite(outFile, o128)
        cv2_imwrite(outFile, o128)
        #n128 = np.asarray(o128).reshape(1,128*128)
        #np.savetxt(f128, n128, "%3d", delimiter=",")
        #ftarget.write(img_path+"\r\n")
        #d = np.array(o64)
        #print(d)
        i+=1
    #sys.exit()


def process(path, dest):
    folders = []
    for root, dirs, files in os.walk(path, topdown=True):
        for folder in dirs:
            if (folder == "."): continue
            if (folder == ".."): continue

            print(path, folder)
            for root, dirs, files in os.walk(path+"/"+folder, topdown=True):
                if not os.path.isdir(dest+"/"+folder):
                    os.mkdir(dest+"/"+folder)
                for file in files:
                    imgPath = path+"/"+folder+"/"+file
                    destPath = dest+"/"+folder+"/"+file
                    extract_img2(imgPath, destPath)

"""            
#img_path = "img1.jpg"
l = glob.glob(sys.argv[1]+"/*.jpg")
for f in l:
    extract_img(f)
"""
#extract_img2("faces\\강혜원\\0223.jpg", "hahaha.jpg")
#extract_img2("0223.jpg", "hahaha.jpg")
#extract_img("/Users/chpark/Documents/CU\ Boulder/2021_Fall/Machine\ Learning/Project/Face-Recognition/extracted_frames/test_frames/00000003.jpg")
#extract_img2("00000081.jpg", "test.jpg")
process("faces", "output")
ftarget.close()

