import glob
import os
import numpy as np

x = np.loadtxt("data/FIRE/Ground Truth/control_points_S71_1_2.txt")
print(x)
num = 0
print(x[:, 0:2])


# for file in glob.glob("./data/FIRE/Images/*.jpg"):
#     print(file)
#     filename = os.path.basename(file)
#     sample_id = filename.split("_")[0]
#     sample_num = filename.split(".")[0].split("_")[1]
#     label = f"control_points_{sample_id}_1_2.txt"
#     if not file == label:
#         print(file, label)