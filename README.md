### Project Title:
**Pedestrian Detection with HOG and SVM Using OpenCV**

---

### README.md

```markdown
# **Pedestrian Detection with HOG and SVM Using OpenCV**

## **Project Overview**
This project implements a pedestrian detection system using the Histogram of Oriented Gradients (HOG) descriptor along with a pre-trained Support Vector Machine (SVM) classifier provided by OpenCV. The system processes an input image and detects pedestrians by drawing bounding boxes around them, both before and after applying Non-Maximum Suppression (NMS) to remove redundant boxes.

Pedestrian detection is a crucial task in computer vision applications such as surveillance systems, autonomous driving, and human-computer interaction. This project showcases how to use OpenCV’s powerful tools to achieve real-time pedestrian detection in images.

---

## **Key Features**
- **Real-Time Pedestrian Detection**: Detects pedestrians in an image using a pre-trained SVM model with HOG features.
- **Non-Maximum Suppression**: Filters out overlapping bounding boxes to ensure that each detected pedestrian is marked only once.
- **Visualization**: Visualizes both the raw detections and the final results after Non-Maximum Suppression with bounding boxes drawn around pedestrians.
- **Efficiency**: Utilizes optimized image processing functions from OpenCV for fast detection and display.

---

## **Technical Details**
### **HOG (Histogram of Oriented Gradients)**
HOG is a feature descriptor used for object detection. It captures gradient orientation in localized portions of an image and is effective for detecting objects like pedestrians.

### **SVM (Support Vector Machine)**
The pre-trained SVM model provided by OpenCV is trained on a large dataset of positive (pedestrian) and negative (non-pedestrian) images to classify regions of an image.

### **Non-Maximum Suppression (NMS)**
Non-Maximum Suppression is applied to remove redundant bounding boxes. If multiple boxes overlap, only the one with the highest confidence score is retained, improving detection accuracy.

---

## **Prerequisites**
To run this project, you will need:
- Python 3.x
- OpenCV (4.x or higher)
- NumPy
- imutils (for resizing and NMS)

---

## **Installation**

### 1. Clone the Repository
To get started, clone the repository to your local machine:
```bash
git clone https://github.com/your-username/pedestrian-detection-hog-svm.git
```

### 2. Install Dependencies
Navigate to the project directory and install the necessary dependencies:
```bash
pip install opencv-python numpy imutils
```

---

## **Usage Instructions**

### Running the Pedestrian Detection
Once you have installed the dependencies, you can run the pedestrian detection script. Ensure that you have an image named `f.jpg` in the same directory or modify the code to load your preferred image.

```bash
python pedestrian_detector.ipynb
```

This script will:
1. Read the input image.
2. Detect pedestrians using the HOG descriptor and the SVM detector.
3. Apply Non-Maximum Suppression to filter overlapping detections.
4. Display two images: one before suppression and one after suppression, both with bounding boxes around detected pedestrians.

### Customizing Input Image
To use your own image, replace the line in the code that loads the image:
```python
img = cv2.imread('your-image.jpg')
```

---

## **Understanding the Code**

### **1. Loading the Image**
The script starts by loading an image from the local directory and resizing it for efficient processing.

```python
img = cv2.imread('f.jpg')
img = resize(img, height=500)
```

### **2. HOG Descriptor Initialization**
The HOG descriptor is initialized with a pre-trained SVM model for detecting pedestrians.

```python
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
```

### **3. Pedestrian Detection**
The `detectMultiScale` function scans the image for pedestrians, returning bounding boxes for each detected pedestrian.

```python
rects, weights = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
```

### **4. Visualizing Results Before Suppression**
The detections are visualized by drawing red rectangles around detected pedestrians before applying Non-Maximum Suppression.

```python
for x, y, w, h in rects:
    cv2.rectangle(copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
cv2.imshow('before suppression', copy)
cv2.waitKey(0)
```

### **5. Applying Non-Maximum Suppression (NMS)**
To eliminate overlapping bounding boxes, NMS is applied. This step ensures that only one bounding box is retained for each detected pedestrian.

```python
pick = non_max_suppression(np.array([[x, y, x+w, y+h] for x, y, w, h in rects]), overlapThresh=0.65)
```

### **6. Visualizing Final Results**
After suppression, the refined detections are drawn with green rectangles.

```python
for xa, ya, xb, yb in pick:
    cv2.rectangle(img, (xa, ya), (xb, yb), (0, 255, 0), 2)
cv2.imshow('after suppression', img)
cv2.waitKey(0)
```

---

## **Example Output**
You will see two windows pop up when running the code:
1. **Before Suppression**: Displays the image with red rectangles around all detected pedestrians, including overlapping boxes.
2. **After Suppression**: Displays the image with green rectangles around the final pedestrian detections, with overlapping boxes removed.


---

## **Enhancements and Future Work**
This pedestrian detection project can be extended in several ways:
1. **Real-Time Detection**: Integrate with live video feeds or camera streams for real-time pedestrian detection.
2. **Deep Learning**: Incorporate deep learning techniques, such as a pre-trained Convolutional Neural Network (CNN), for more accurate and faster detection.
3. **Multiple Object Detection**: Add functionality to detect other objects such as vehicles, bicycles, or animals.

---

## **Troubleshooting**

- **No pedestrians detected?** Make sure the input image contains clear pedestrians and that there is sufficient contrast between the pedestrians and the background.
- **Slow performance?** Resize the image to a smaller dimension to improve processing speed, or run the detection on a more powerful machine.
- **Errors while loading image?** Ensure that the file path to the input image is correct and that the file exists in the same directory as the code.

---

## **References**
- [OpenCV Documentation](https://docs.opencv.org/)
- [HOG Descriptor in OpenCV](https://docs.opencv.org/3.4/da/ded/tutorial_hog.html)
- [Pedestrian Detection with HOG and SVM](https://learnopencv.com/histogram-of-oriented-gradients/)

---

## **Acknowledgements**
- **OpenCV Library**: A huge thanks to the OpenCV team for developing and maintaining this amazing library.
- **imutils Library**: The `imutils` package is a simple but useful library for basic image processing operations.
- **Non-Maximum Suppression Algorithm**: Thanks to the developers who contributed this algorithm for object detection and computer vision applications.

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

This README provides a detailed explanation of the project, step-by-step usage instructions, and technical details of the pedestrian detection process. You can personalize the project further by including example outputs and additional customizations!
