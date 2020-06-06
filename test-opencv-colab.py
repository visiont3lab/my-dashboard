from matplotlib import pyplot as plt
import cv2

dir_path = "assets/images/"
names = os.listdir(dir_path)

for name in names:
    path = os.path.join(dir_path,name) #dir_path +
    img = cv2.imread(path,1) # bgr , rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB ) 
  
    plt.figure(figsize=(18,8))
    plt.imshow(img)
    plt.show()
