# CSST106-CS4B-MP2

### Machine Problem No. 2: Applying Image Processing Techniques

Scaling and rotation are common image transformations in computer vision, often used for resizing images or rotating them to a specific angle. Using OpenCV, you can easily apply these transformations.

#### Scaling (Resizing)
Scaling changes the size of an image. In OpenCV, you can use the `cv2.resize()` function.

```python
import cv2

# Load an image
image = cv2.imread('image.jpg')

# Scale the image by 50%
scaled_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# Show the scaled image
cv2.imshow('Scaled Image', scaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

fx and fy are the scaling factors along the x and y axes, respectively.
interpolation is the method used for resizing; cv2.INTER_LINEAR is a common choice.

#### Rotation

Rotation rotates the image around a specified point. You use the cv2.getRotationMatrix2D() function to get the rotation matrix and then cv2.warpAffine() to apply the transformation.

```python
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
```
The 45 is the angle of rotation in degrees (positive for counterclockwise).
The 1.0 is the scaling factor; 1.0 means no scaling.


# Implementation 
 Install OpenCV 
 ! pip install opencv-python-headless

 ### Import Libraries 
 ``` python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

### Image Processing 
``` python
def display_image(img,title="Image"):
  plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  plt.title(title)
  plt.axis('off')
  plt.show()

def display_images(img1, img2, title1 = "Image 1", title2="Image2"):
  plt.subplot(1,2,1)
  plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
  plt.title(title1)
  plt.axis('off')

  plt.subplot(1,2,2)
  plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
  plt.title(title2)
  plt.axis('off')

  plt.show()
```

### Load image 
``` python
from google.colab import files
from io import BytesIO
from PIL import Image

uploaded = files.upload()
image_path = next(iter(uploaded))
image = Image.open(BytesIO(uploaded[image_path]))
image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)


display_image(image, "Original Image")
```
![nemo](https://github.com/user-attachments/assets/197e022e-cded-43b3-ae49-5629f71eabbc)


### Exercise 1: Scaling and Rotation 
``` python
def scale_image(img, scale_factor):
  height, width = img.shape[:2]
  scale_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation = cv2.INTER_LINEAR)
  return scale_img

def rotate_image(image, angle):
  height, width = image.shape[:2]
  center = (width//2,height//2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1)
  rotated_image = cv2.warpAffine(image, matrix,(width, height))
  return rotated_image

scaled_image = scale_image(image, 0.5)
display_image(scaled_image, "Scaled Image")

rotated_image = rotate_image(image, 45)
display_image(rotated_image, "Rotated Image")
```
![copy](https://github.com/user-attachments/assets/f11542dd-c5d9-4a32-b96d-d18c32f63d7a)


### Exercise 2: Blurring  Techniques
``` python
gaussian_blur = cv2.GaussianBlur(image, (35,35),0)
display_image(gaussian_blur, "Gaussian Blur")

median_blur = cv2.medianBlur(image,15)
display_image(median_blur, "Median Blur")

bilateral_filter = cv2.bilateralFilter(image,25, 85, 85)
display_image(bilateral_filter, "Bilateral Filter")
```
![copy2](https://github.com/user-attachments/assets/8e218270-44f2-4319-a9f8-f86ece6ebf50)


### Exercise 3: Edge Detection using Canny 
``` python
edge = cv2.Canny(image, 100, 200)
display_image(edge, "Edge Detection")
```
![copy3](https://github.com/user-attachments/assets/627773e8-79e1-4c4c-ba8f-136a6573f8df)

### Exercise 4: Basic Image Process (Interactive)
``` python
 def scale_image(img, scale_factor):
    height, width = img.shape[:2]
    return cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (width, height))

def process_image(img, actions):
    processed_img = img.copy()
    
    for action in actions:
       
        action = action.lower().strip()
        
        if action == 'scale':
            processed_img = scale_image(processed_img, 0.5)
        elif action == 'rotate':
            processed_img = rotate_image(processed_img, 45)
        elif action == 'gaussian_blur':
            processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        elif action == 'median_blur':
            processed_img = cv2.medianBlur(processed_img, 5)
        elif action == 'canny':
            processed_img = cv2.Canny(processed_img, 100, 200)
    
    return processed_img

def display_images(original, processed, original_title, processed_title):
    plt.figure(figsize=(10, 5))

   # Show Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(original_title)
    plt.axis('off')

    # Show Process Image 
    plt.subplot(1, 2, 2)
    plt.title(processed_title)
    if processed.ndim == 2:
        plt.imshow(processed, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
# Assuming `image` is already loaded
action_input = input("Enter actions (e.g., scale, rotate, gaussian_blur, median_blur, canny) separated by commas: ")
actions = action_input.split(',')

processed_image = process_image(image, actions)
display_images(image, processed_image, "Original Image", f"Processed Image ({', '.join(actions)})")

```


### Exercise 5: Comparison Of Filtering Techniques

``` python
# Applying Gaussian, Median, and Bilateral filters
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
median_blur = cv2.medianBlur(image, 5)
bilateral_filter = cv2.bilateralFilter(image, 9, 75, 75)
"""
cv2.bilateralFilter(): This filter smooths the image while keeping edges sharp, unlike
Gaussian or median filters. It’s useful for reducing noise while preserving details.
"""
# Display the results for comparison
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title("Gaussian Blur")
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title("Median Blur")
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB))
plt.title("Bilateral Filter")
plt.show()
```
