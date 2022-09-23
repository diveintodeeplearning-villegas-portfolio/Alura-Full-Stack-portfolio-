import cv2
import numpy as np
try:
 from PIL import Image
except ImportError:
 import Image
import pytesseract
import matplotlib.pyplot as plt 
import os
from ImageCropperPNG import ImageCropper
import PySimpleGUI as sg


class LowerBoundDetector:
  def __init__(self, image_dir):
    self.image_dir = image_dir
    self.image     = cv2.imread(image_dir)
    self.graph_window = None
    self.graph_image  = None
    self.lowerbound_image = None
    self.cnts = None
    self.stockcode_window = None
    self.stockcode_image = None
    self.stockcode_num = None

    
  def detect_graph_stock_windows(self):
    self.image = cv2.resize(self.image, (1000,700), interpolation = cv2.INTER_AREA)
    image_org = self.image.copy()

    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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
              cv2.rectangle(self.image, (x, y), (x + w, y + h), (36,255,12), 2)
              cv2.rectangle(self.image, (x, y), (x+130, y+50), (36,0,12), 2)
              image_number += 1
              #print(cv2.boundingRect(c))

    #cv2_imshow( sharpen)
    #cv2_imshow(close)
    #cv2_imshow(thresh)
    #cv2_imshow(image)
    cv2.imshow('graph',graph)
    cv2.imshow('stock',stockCode)
    cv2.waitKey()
    #cv2.waitKey()
    self.graph_window = graph
    self.stockcode_window = stockCode

    return (graph, stockCode)


  def detect_stockcode_window(self,image):
    stockcode=[]
    #image = self.stockcode_window
    image_org = image.copy()#self.stockcode_window.copy()

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
    self.stockcode_image = stockcode
    return stockcode



  def get_stockcode(self,img):
    #shape = img.shape
    #img = cv2.resize(img, (shape[0],shape[1]), interpolation = cv2.INTER_AREA)
    #img = self.stockcode_image
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
    self.stockcode_num = num

    #cv2_imshow(img)
    return num

  
  def detect_graph_window(self,image):
      ''' function Detecting Edges '''
      #image = self.graph_window 
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
          self.graph_image = img
          return img


      

  def detect_lower_bound_graph(self,image): 
      #image = self.graph_image
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
            cv2.imshow(crop_lowerbound_img)
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
            self.lowerbound_image = crop_lowerbound_img
            return crop_lowerbound_img


  def detect_contours(self,image): 
      #image = self.lowerbound_image
      image_with_edges = cv2.Canny( image , 500, 700)
      #cv2_imshow(image_with_edges)
      cnts = cv2.findContours(image_with_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_LIST
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      self.cnts = cnts
      return cnts

  def createWindowViewer(self,image):
	  layout = [[sg.Text("Check stock code image")], [sg.Image(image)],[sg.Button("OK")]]
	  #key="-IMAGE-"

	  # Create the window
	  window = sg.Window("Demo", layout)
	  # Create an event loop
	  while True:
	      event, values = window.read()
	      #window["-IMAGE-"].update(filename='myhero4_cropped.png')
	      # End program if user closes window or
	      # presses the OK button
	      if event == "OK" or event == sg.WIN_CLOSED:
	          window.close()
	          break
	  window.close()


  def run(self):

      stockcode_num = None

      graph,stock = self.detect_graph_stock_windows()
      

      #cv2.imshow(graph_window)

        #cropper = ImageCropper()

        #cropper.set_file(self.image_dir)

        #cropper.run()

        
	    #self.image     = cv2.imread(image_dir)
	    #self.graph_window = None
	    #self.graph_image  = None
	    #self.lowerbound_image = None
	    #self.cnts = None
	    #self.stockcode_window = None
	    #self.stockcode_image = None
	    #self.stockcode_num = None
	      

      mygraph=self.detect_graph_window(graph)

      lower=self.detect_lower_bound_graph(mygraph)

      cnts = self.detect_contours(lower)


      if cnts: #self.cnts:
        #cv2_imshow(stockcode_window)
        stockcode_image = self.detect_stockcode_window(stock)

        stockcode_num = self.get_stockcode(stockcode_image)

      return self.stockcode_num


        