from ximea import xiapi
import os
import cv2
import numpy as np
export_folder = "Ximea_captures"
filename = "testing" + ".png"
class Ximea: 
    def __init__(self):
        self.cam = xiapi.Camera()
        self.cam.open_device()
        #settings
        self.cam.set_exposure(1000)
        self.cam.set_exp_priority(1) #1 = exposure priority compared to gain
        gain = self.cam.get_gain()
        print( f"Exposure: {self.cam.get_exposure()} us")
        print( f"Gain: {gain} dB")
    def capture(self,export_folder,filename):
        img = xiapi.Image()
        self.cam.start_acquisition()
        self.cam.get_image(img)
        img_np = img.get_image_data_numpy()   
        img_gray = np.asarray(img_np * 255, dtype=np.uint8)
        fn_save = os.path.join(export_folder, filename)
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)

        if cv2.imwrite(fn_save, img_gray):
            print(f"Saved {fn_save}")
        else:
            print(f"Failed to save {fn_save}")
        #stop data acquisition
        self.cam.stop_acquisition()
        #stop communication
        self.cam.close_device()

Ximeas = Ximea()

Ximeas.capture(export_folder,filename)