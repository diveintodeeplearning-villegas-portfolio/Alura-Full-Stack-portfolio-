from ImageCropperPNG import ImageCropper
#from LowerBoundDetector import LowerBoundDetector
import os
from LowerBoundDetector_utils import detectlowerbound



DIR='tmp_images'
img_name='myhero'
images_list = os.listdir(DIR)
images_list = [i for i in images_list if i.startswith(img_name)]
print(images_list)

LB_stockcodes_list=[]
'''
for i in [images_list[0]]:
  sc_num = detectlowerbound(DIR+'/'+i)
  if sc_num:
    LB_stockcodes_list.append(sc_num)

print(LB_stockcodes_list)
'''


for i in images_list:
	cropper = ImageCropper()

	cropper.set_file(DIR+'/'+i)

	cropper.run()


'''
for i in images_list:
  image = cv2.imread('tmp_images/'+i)
  graph_window, stockcode_window = detect_graph_stock_windows(image)

  cv2_imshow(graph_window)
  graph_image = detect_graph_window(graph_window)

  lowerbound_image = detect_lower_bound_graph(graph_image)

  cnts = detect_contours(lowerbound_image)

  if cnts:
    cv2_imshow(stockcode_window)
    stockcode_image = detect_stockcode_window(stockcode_window)

    stockcode_num = get_stockcode(stockcode_image)

    LB_stockcodes_list.append(stockcode_num)

print('Lower bound stock code list:')
print(LB_stockcodes_list)
'''
'''
lb = LowerBoundDetector(DIR+'/'+images_list[0])
tmp_stock_code = lb.run()

if tmp_stock_code:
  LB_stockcodes_list.append(tmp_stock_code)
print(LB_stockcodes_list)
'''
