import os
import numpy as np

image_files = []
os.chdir(os.path.join("data", "obj"))
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".JPG"):
        image_files.append("data/obj/" + filename)
os.chdir("..")
#Split train 80% - validate 10% - test 10%
#Primo metodo
np.random.shuffle(image_files)
training, validation, testing = np.split(image_files, [int(len(image_files)*0.8), int(len(image_files)*0.9)])
#Secondo metodo
#training = image_files[:int(len(image_files)*0.8)] 
#validation = image_files[int(len(image_files)*0.8):int(len(image_files)*0.9)]
#testing = image_files[int(len(image_files)*0.9):]


with open("train.txt", "w") as outfile:
    for image in training:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
with open("valid.txt", "w") as outfile:
    for image in validation:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
with open("test.txt", "w") as outfile:
    for image in testing:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()
os.chdir("..")

