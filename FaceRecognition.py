import numpy as np
import cv2
import math
import matplotlib

rows = 112
cols = 92
len_row = rows*cols

# ------Creating the DATA MATRIX ----------

data_matrix = np.zeros((len_row,280),np.uint8)
col_pointer = 0
# Taking all the images from the training set and creating the data matrix by keeping each sample as a column vector

for j in range(1,41):
    for i in range(1,8):
        s = "ATT/s"+str(j)+"/"+str(i)+".pgm"
        im = cv2.imread(str(s),cv2.IMREAD_GRAYSCALE)
        im=np.reshape(im,(len_row,1))
        #print(im.shape)
        data_matrix[:,col_pointer] = im[:,0]
        col_pointer+=1
#print(col_pointer)

# cv2.imshow('Meanface',np.reshape(data_matrix[:,40],(112,92)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ------ Finding MEAN FACE --------------------

# Taking mean of all the samples to get a Mean face

meanface = np.zeros((len_row,1),np.uint8)
meanface = data_matrix.sum(axis=1)
meanface = np.reshape(meanface,(len_row,1))

meanface = meanface//280
meanface = np.uint8(meanface)

meanface_updated=np.reshape(meanface,(112,92))
meanface_updated = np.uint8(meanface_updated)

# cv2.imwrite("meanface.jpg",meanface_updated)
# cv2.imshow('Meanface',meanface_updated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# ---------- Normalizing the DATA MATRIX ---------
# Subtracting the meanface from all the samples to get the normalized data matrix

data_normalized = data_matrix - meanface

#cv2.imshow('Meanface',np.reshape(data_normalized[:,60],(112,92)))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# --------- TRAINING PHASE (PCA) ---------------

#Finding the covariance matrix
cov_matrix = data_normalized.transpose()//math.sqrt(280)

# Finding SVD to get the principal components
U, s, Vt = np.linalg.svd(cov_matrix, full_matrices=False)

# Taking the first 280 principal components
pc = Vt.transpose()[:,:280]

# We get the reduced data matrix by Y = PX
data_reduced = np.dot(pc.transpose(),data_normalized)

# ------------ TEST PHASE ----------------------

# We take the remaining 120 images to be the test set and find the nearest neighbour of each test set to find the subject of that image.

result = np.zeros(120)
iteration = 0
ground_truth = []
for k in range(1,41):
    for l in range(8,11):
        s = "ATT/s" + str(k) + "/" + str(l) + ".pgm"
        ground_truth.append(k) # Creating the ground truth array
        test = cv2.imread(str(s), cv2.IMREAD_GRAYSCALE)
        test = np.reshape(test, (len_row, 1))

        test_normalized = test - meanface

        test_data = data_reduced - np.dot(pc.transpose(), test_normalized)
        norm_val = np.sqrt(np.sum(test_data * test_data, axis=0))

        related_col = np.argmin(norm_val)
        final_subject = related_col // 7 + 1
        result[iteration] = final_subject # finding the result subject
        iteration+=1

ground_truth = np.asarray(ground_truth)

# -------------- Finding ACCURACY ------------------

# We are finding the accuracy from the result we get and from the ground truth.

accuracy_array = result - ground_truth
count = 0
for i in range(len(accuracy_array)):
    if accuracy_array[i]==0:
        count+=1

accuracy = (count/120)*100
print("accuracy = ",accuracy,"%")

