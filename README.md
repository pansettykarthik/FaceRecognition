# Facial Recognition 
Facial Recognition System using eigenfaces in Python. For the course "Mathematical Foundations for Computer Vision and Graphics Course (ES637)."

The dataset used for Facial recognition is AT&T "The Database of Faces" (formerly "The ORL Database of Faces").
- There are 40 subjects and each subject has 10 images, which totals to 400 images.
- The dimension of each image is 112x92. So, the dimension of the data matrix becomes 10,304x280.
- The test training ratio used is 7:3. So, there are 280 images in the training set and 120 images in the test set.
- The number of principal components used for dimensionality reduction is 280.
- The dimension of the reduced data matrix becomes 280x280.
- The accuracy on the test set of 120 images is 90%. This accuracy comes from the fact that we labeled 108 images of the testset correctly to the subject out of the 120 images.


