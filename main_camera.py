import time
import cv2
import numpy as np

from facedetector import CascadeFaceDetector, OpencvFaceDetector
from config import CascadeFaceDetectorConfig, OpencvFaceDetectorConfig


def main():
    fd = OpencvFaceDetector(OpencvFaceDetectorConfig)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set Width
    cap.set(4, 480)  # set Height

    time.sleep(0.5)

    while True:
        ret, frame = cap.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fd.detect_and_draw_faces(frame)

        cv2.imshow('frame', frame)
        # cv2.imshow('gray', gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()