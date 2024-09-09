# CSST106-CS4B-MP2

### Machine Problem No. 2: Applying Image Processing Techniques

Scaling and rotation are common image transformations in computer vision, often used for resizing images or rotating them to a specific angle. Using OpenCV, you can easily apply these transformations.

Scaling (Resizing)
Scaling changes the size of an image. In OpenCV, you can use the cv2.resize() function.

``python

import cv2

# Load an image
image = cv2.imread('image.jpg')

# Scale the image by 50%
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Show the scaled image
cv2.imshow('Scaled Image', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
fx and fy are the scaling factors along the x and y axes, respectively.
interpolation is the method used for resizing; cv2.INTER_LINEAR is a common choice.
Rotation
Rotation rotates the image around a specified point. You use the cv2.getRotationMatrix2D() function to get the rotation matrix and then cv2.warpAffine() to apply the transformation.

python
Copy code
import cv2

# Load an image
image = cv2.imread('image.jpg')

# Get the image dimensions
(h, w) = image.shape[:2]

# Define the center of the image
center = (w // 2, h // 2)

# Define the rotation matrix: 45-degree rotation, no scaling
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

# Rotate the image
rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

# Show the rotated image
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
The 45 is the angle of rotation in degrees (positive for counterclockwise).
The 1.0 is the scaling factor; 1.0 means no scaling.
