import pytesseract
import cv2
import math
import numpy as np
import re
try:
    from PIL import Image
except ImportError:
    import Image

from invoice2data import extract_data
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize


path= 'C://1.jpg'

img = cv2.imread(path)
img=cv2.resize(img,(896,896))
img_h = cv2.imread(path)
img_h= cv2.resize(img_h,(896,896))
img_c = cv2.imread(path)
img_c= cv2.resize(img_c,(896,896))
cv2.namedWindow('Thresholded',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Thresholded', 600, 600)

n = 1 # N. . .
p,q= img.shape[:2]

for i in range (int(0.27*p),p):
    for j in range(0,q):
           img_h[i,j]=0

for i in range (0,int(0.27*p)):
    for j in range(0,q):
           img_c[i,j]=0



org_imgh = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)
org_imgc= cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
org_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(org_imgh)
cl2 = clahe.apply(org_imgc)
cl3 = clahe.apply(org_img)


# ret, thi = cv2.threshold(cl3, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _,contoursh,_ = cv2.findContours(thi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for contour in contoursh:
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     (x, y), (w, h), angle = rect
#     print(x,y)
#     box = np.int0(box)
#     perimeter = cv2.arcLength(contour, True)
#     color_map = (int(np.random.randint(0, 255, 1)[0]),
#                  int(np.random.randint(0, 255, 1)[0]),
#                  int(np.random.randint(0, 255, 1)[0]))
#
#
#     if perimeter>500:
#         img = cv2.drawContours(img, [box], 0, color_map, 2)
#
#         M = cv2.moments(contour)
#
#     #Cx = int(M["m10"] / M["m00"])
#     #Cy = int(M["m01"] / M["m00"])
#     #print(M)
#     #cv2.circle(img, (Cx, Cy), 7, (255, 255, 255), -1)
#     #cv2.putText(img, "center", (Cx - 20, Cy - 20),
#      #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         print(angle)
#
#
# cv2.imshow("thresholded",img)
#
# v = cv2.waitKey(0)
# cv2.destroyAllWindows()




cv2.imwrite("header.png",cl1)
header_text = pytesseract.image_to_string(Image.open("header.png"))
print(header_text)

#content_text = pytesseract.image_to_string(Image.open("content.png"))
#print(content_text)

ex = header_text
ex= ex.replace('\n', '. ')
words = word_tokenize(ex)
sentence = sent_tokenize(ex)

#sentence = [s.replace('\n', '.') for s in sentence]

print(sentence)
#print(word_tokenize(ex))
#print(sentence[0])

# #shop_name_idt = {'LTD', 'Limited', 'Linited', 'LIMITED'}
# addr_idt =

# for i in len(words):
#     if words[i]
#result = extract_data(text)
#print(result)
filename='C:\City_dataset.txt'

chr=[',']
lineList = [line.rstrip('\n') for line in open(filename)]
chr2=['-']
def common(a,b):
    c = [value for value in a if value in b]
    return c

print("hey")
stopwords = nltk.corpus.stopwords.words('english')
#print(stopwords)


address=[]
names=[]
for i in range(0, len(sentence)):


    words = word_tokenize(sentence[i])
    #print(words)
    e = common(words, lineList)
    d = common(words, chr)
    f = common(words, chr2)
    if e!=[]:
        if d!=[]:
            address.append(sentence[i])
        else:
            #if no comma but still place, check if pincode is a part of the string
            pins=re.findall(r"[0-9]{6} |[0-9]{3}\s[0-9]{3}",sentence[i])
            if pins != []:
                address.append(sentence[i])

            else:
                names.append(sentence[i])
    else:
        if d!=[]:
            address.append(sentence[i])
        else:
            print("None bro.")
print(address)
print(names)
stopwords.append(address)
print(stopwords)