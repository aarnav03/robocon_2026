import cv2
import numpy as np

cap = cv2.VideoCapture(0)   # change index if needed

if not cap.isOpened():
    print("❌ Camera not opened")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1,mask2)

    symbol_mask = cv2.bitwise_not(red_mask) #invert

    kernel = np.ones((3,3), np.uint8)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_OPEN, kernel)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        symbol_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    # edge detection

    symbol_mask = symbol_mask.astype(np.uint8)

    blur_sym = cv2.GaussianBlur(symbol_mask, (3,3), 0)

    edges = cv2.Canny(
        blur_sym,
        threshold1=50,
        threshold2=150
    )


    output = frame.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area < 800:     #tunable thingie 
            continue

        x,y,w,h = cv2.boundingRect(c)

        #bounding box
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)

        roi = symbol_mask[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64,64))
        cv2.imshow("ROI", roi)

    cv2.imshow("Camera", frame)
    cv2.imshow("Symbol Mask", symbol_mask)
    cv2.imshow("Detected Symbols", output)
    cv2.imshow("edges",edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()

