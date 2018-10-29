import numpy
import time
from PIL import Image
from darknet.api2.predictor import get_YOLOv3_Predictor, get_YOLOv3_Tiny_Predictor

filename = "dog.jpg"
image = Image.open(filename)
img = numpy.array(image)

# this will take few seconds .. but we need to create predictor only once
predictor = get_YOLOv3_Predictor()
# predictor = get_YOLOv3_Tiny_Predictor()

t = time.time()
lis = predictor(img)
print("Predicting took", time.time()-t, "seconds") # takes around 0.2 secs on a decent (5.1 grade) GPU
for l in lis:
    print(l)
