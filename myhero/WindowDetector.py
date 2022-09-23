import cv2
import numpy as np
try:
 from PIL import Image
except ImportError:
 import Image
import pytesseract
import matplotlib.pyplot as plt 
import os

class GraphWindow:
  def __init__(self, image):
    self.image = image


  def detect_graph_stock_windows(self):
    self.image = cv2.resize(self.image, (1000,700), interpolation = cv2.INTER_AREA)
    image_org = self.image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,100,255, cv2.THRESH_BINARY_INV)[1]#80#100

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 1000
    max_area = 150000000
    image_number = 0

    # x,y,w,h = (477, 81, 490, 505)
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            if h > 450 and w<600:
              graph = image_org[y:y+h, x:x+w]
              stockCode = image_org[y:y+50, x:x+130]
              cv2.imwrite('myGraphWindow_{}.png'.format(image_number), graph)
              cv2.imwrite('myStockCode_{}.png'.format(image_number), stockCode)
              cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
              cv2.rectangle(image, (x, y), (x+130, y+50), (36,0,12), 2)
              image_number += 1
              #print(cv2.boundingRect(c))

    #cv2_imshow( sharpen)
    #cv2_imshow(close)
    #cv2_imshow(thresh)
    #cv2_imshow(image)
    #cv2.waitKey()
    return (graph, stockCode)


  def detect_stockcode_window(image):
    stockcode=[]
    image_org = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.threshold(sharpen,170,255, cv2.THRESH_BINARY_INV)[1]#150

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    min_area = 1
    max_area = 1500
    image_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            if w > 50:
              stockcode = image_org[y:y+h, x:x+w]
              cv2.imwrite('mySCnumber_{}.png'.format(image_number), stockcode)
              cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
              image_number += 1
              #print(cv2.boundingRect(c))
    #cv2_imshow( sharpen)
    #cv2_imshow(close)
    #cv2_imshow(thresh)
    #cv2_imshow(image)
    #cv2.waitKey()
    return stockcode



  def get_stockcode(img):
    #shape = img.shape
    #img = cv2.resize(img, (shape[0],shape[1]), interpolation = cv2.INTER_AREA)

    img =img[:,65:]
    #gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thr = cv2.adaptiveThreshold(gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                            cv2.THRESH_BINARY_INV, 21, 9)
    
    txt = pytesseract.image_to_string(img, config='--psm 6')
    #print(txt)

    num=''
    for i in txt:
      if i.isdigit():
        num +=i
    #print(num)

    #cv2_imshow(img)
    return num

  
  def detect_graph_window(image):
      ''' function Detecting Edges '''

      image_copy = image.copy()

      image_with_edges = cv2.Canny(image , 20, 600, L2gradient = True)

      images = [image , image_with_edges]
      

      location = [121, 122]
      
      
      cnts = cv2.findContours(image_with_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]

      min_area=40000
      max_area=10000000

      image_number=0
      for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
          x,y,w,h = cv2.boundingRect(c)
          img = image_copy[y:y+h,x:x+w]
          cv2.rectangle(image_copy, (x, y), (x + w, y + h), (36,255,12), 2)
          #print( cv2.boundingRect(c))
          cv2.imwrite('mygraph_{}.png'.format(image_number), img)
          image_number += 1
          return img
      

  def detect_lower_bound_graph(image): 
      
      image_with_edges = cv2.Canny( image , 500, 700)

      cnts = cv2.findContours(image_with_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_LIST
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]

      min_area=1
      max_area=1000000000

      image_number=0

      images = [image , image_with_edges]

      location = [121, 122]

      line_thickness = 1


      for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
          x,y,w,h = cv2.boundingRect(c)
          #print(cv2.boundingRect(c))
          if w >400:
            img = image[y+1:y+h,x:x+w]
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,0), 2)
            #cv2.line(image, (x, y+int(h/2)), (x+w, y+int(h/2)), (0, 0, 0), thickness=line_thickness)
            #cv2.line(image, (x+int(w*(2/3)), y), (x+int(w*(2/3)), y+h), (0, 0, 0), thickness=line_thickness)
            #cv2.line(image, (x, y+h), (x+w, y+h), (0, 0, 0), thickness=line_thickness)
            crop_lowerbound_img0 = image[y+int(h/2):y+h, x+int(w*(2/3)):x+w]
            crop_lowerbound_img = image[y+int(h/2):y+h, x+int(w*(2/3)):x+w-55]
            #cv2_imshow( crop_lowerbound_img0)
            cv2.imwrite('myRecentLowerBound_{}.png'.format(image_number), crop_lowerbound_img)
            #cv2.imwrite('mygraphQsc_{}.png'.format(image_number), image)
            #imge = image_with_edges[y+1:y+h,x:x+w]
            #cv2.rectangle(imge, (x, y), (x + w, y + h), (36,255,12), 2)
            #cv2.line(imge, (x, y+int(h/2)), (x+w, y+int(h/2)), (255, 255, 255), thickness=line_thickness)
            #cv2.line(imge, (x+int(w*(2/3)), y), (x+int(w*(2/3)), y+h), (255, 255, 255), thickness=line_thickness)
            #cv2.line(imge, (x, y+h), (x+w, y+h), (255, 255, 255), thickness=line_thickness)

           
            #crop_img = imge[y+int(h/2):y+h, x+int(w*(2/3)):x+w]
            #cv2_imshow( crop_img)

            #print( cv2.boundingRect(c))
            #cv2.imwrite('mygraphQs_{}.png'.format(image_number), imge)
            image_number += 1
            return crop_lowerbound_img


  def detect_contours(image): 
      image_with_edges = cv2.Canny( image , 500, 700)
      #cv2_imshow(image_with_edges)
      cnts = cv2.findContours(image_with_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_LIST
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      return cnts

  def run(self):

      stockcode_num = None

      graph_window, stockcode_window = detect_graph_stock_windows(self)

      #cv2.imshow(graph_window)
      graph_image = detect_graph_window(graph_window)

      lowerbound_image = detect_lower_bound_graph(graph_image)

      cnts = detect_contours(lowerbound_image)

      if cnts:
        #cv2_imshow(stockcode_window)
        stockcode_image = detect_stockcode_window(stockcode_window)

        stockcode_num = get_stockcode(stockcode_image)

      return stockcode_num


        