from ximea import xiapi
import os
import cv2
import numpy as np
import time

class Ximea: 
    def __init__(self,export_folder= "",filename= ""):
        self.cam = xiapi.Camera()
        #try:
        self.cam.open_device()  # Open the camera
        self.cam.start_acquisition()  # Start capturing
        #except xiapi.Xi_error as e:
        #    print(f"Failed to initialize camera: {e}")
        #    exit(1)  # Exit if camera cannot be opened
        self.img = xiapi.Image()
        timestr = time.strftime("%Y%m%d")
        cwd = os.path.abspath(os.path.dirname(__file__))

        self.export_folder = os.path.abspath(os.path.join(cwd, "../"))
        self.export_folder  = os.path.abspath(os.path.join(self.export_folder,export_folder))
 

        self.filename = filename + "Ximea"
        self.filename_date = timestr + self.filename
    def set_param(self,exposure=1000):
        #settings
        self.cam.set_exposure(exposure)
        self.cam.set_exp_priority(1) #1 = exposure priority compared to gain
        gain = self.cam.get_gain()
        print( f"Exposure: {self.cam.get_exposure()} us")
        print( f"Gain: {gain} dB")

    def live_view(self,before=True):
        key_pressed = None

        while key_pressed != "q" or key_pressed != "s":
            self.cam.get_image(self.img)
            self.img_np = self.img.get_image_data_numpy()
            cv2.imshow("Main", self.img_np)

            key_pressed = None
            # fmt: off
            # Listen for keypresses from within OpenCV plot
            cv2_key = cv2.waitKey(1)
            if cv2_key == ord("q"):   key_pressed = "q"
            if cv2_key == ord("s"):   key_pressed = "s"
            if key_pressed== "q":
                print("Data not saved")
                break
            if key_pressed =="s":
                if before==True:
                    self.filename_str = "before"
                else: 
                    self.filename_str = "after"
                self.capture()
            
                break
        self.stop()
    def capture(self):
        img_gray = np.asarray(self.img_np * 255, dtype=np.uint8)
        self.filename_date  =self.filename_date + self.filename_str + ".png"
        fn_save = os.path.join(self.export_folder, self.filename_date)
        if not os.path.exists(self.export_folder):
            os.makedirs(self.export_folder)

        if cv2.imwrite(fn_save, img_gray):
            print(f"Saved {fn_save}")
        else:
            print(f"Failed to save {fn_save}")
        #self.stop()

    def stop(self):
        #stop data acquisition
        self.cam.stop_acquisition()
        #stop communication
        self.cam.close_device()
        cv2.destroyAllWindows()





