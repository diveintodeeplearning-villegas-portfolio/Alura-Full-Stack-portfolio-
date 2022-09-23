import cv2
import math
def detect_lowerbound(imagepath):
    # Load image, convert to grayscale, Otsu's threshold
    image = cv2.imread(imagepath)
    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #Detect graph line
    for c in cnts:
      line = [item for sublist in c for item in sublist]
      if len(line)==4:
        myline=line
        cv2.drawContours(result, [cnts[6]], -1, (36,255,12), 2)
        #print(line[0][1])
    y=0
    x=0
    h,w,c = image.shape
    h2=myline[0][1]

    #crop lower bound area
    img = image[y+1:y+h,x:x+w]
    #cv2.rectangle(image, (x, y), (x + w, y + h2), (0,0,0), 2)
    #cv2.line(image, (x, y+int(h2/2)), (x+w, y+int(h2/2)), (0, 0, 0), thickness=2)
    #cv2.line(image, (x+int(w*(2/3)), y), (x+int(w*(2/3)), y+h2), (0, 0, 0), thickness=2)
    #cv2.line(image, (x, y+h2), (x+w, y+h2), (0, 0, 0), thickness=2)
    #cv2_imshow( image)
    crop_lowerbound_img = image[y+int(h2/2):y+h2, x+int(w*(2/3)):x+w]
    cv2_imshow( crop_lowerbound_img)
    cv2.imwrite('mylb.png', crop_lowerbound_img)
    #cv2_imshow(result)
    return crop_lowerbound_img

def detect_contours(image): 
    image_with_edges = cv2.Canny( image , 500, 700)
    cv2_imshow(image_with_edges)
    cnts = cv2.findContours(image_with_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_LIST
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts
        
#image_graph = cv2.imread('mylb.png')
#cv2_imshow(image_graph)
lb=detect_lowerbound('myhero4_copy.png')
cnts= detect_contours(lb)

if cnts :
  print('It is in the lower bound area ')
else:
  print('It is not in the lower bound area')

