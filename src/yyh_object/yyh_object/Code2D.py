import cv2
import  numpy as np
from pyzbar import pyzbar
from Coord_Trans import locate_d4
#飞机识别二维码
def Code2D(param):

    center = None
    barcodes = pyzbar.decode(param.d435i_color) 
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        (x, y, w, h) = barcode.rect
        cv2.rectangle(param.d435i_color, (x, y), (x + w, y + h), (0, 255, 0), 2)#在图中画出来
        center = [x+w/2,y+h/2,x,y,w,h,1,2]#kind = 2代表2维码
        #todo 需要有更多筛选二维码的逻辑
        return locate_d4(param,center)#只返回第一个

      

    
  
    

