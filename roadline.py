import matplotlib.pyplot as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0,255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height*0.5),
        (width*0.7, height*0.5),
        (width*0.7, height),
        (0, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5,5), 0)

    canny_image = cv2.Canny(blurred_image, 50, 150)

    cropped_image = region_of_interest(canny_image,
                                      np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=20,
                            minLineLength=40,
                            maxLineGap=100 )

    
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('C:/Users/Dell/Downloads/test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = process(frame)
        cv2.imshow('frame', frame) 
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()