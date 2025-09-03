import cv2

cap = cv2.VideoCapture(0)  # try built-in camera
if not cap.isOpened():
    print("❌ Cannot open camera with index 0")
else:
    print("✅ Camera index 0 works")
cap.release()
