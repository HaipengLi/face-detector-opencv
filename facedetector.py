import cv2
import numpy as np


class CascadeFaceDetector:
    def __init__(self, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(self.config.classifier_path)

    def detect_faces(self, img):
        """
        @img: np array with bgr color
        """
        # cvt to gray & detect
        if len(img.shape) < 3:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size
        )
        return faces

    @classmethod
    def draw_faces(cls, img, faces, color=(255, 0, 0)):
        if len(img.shape) < 3:
            color = 255
        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    def detect_and_draw_faces(self, img):
        faces = self.detect_faces(img)
        self.draw_faces(img, faces)


class OpencvFaceDetector:
    def __init__(self, config):
        self.config = config
        self.net = cv2.dnn.readNetFromCaffe(self.config.prototxt_path, self.config.model_weights_path)

    def detect_faces(self, img):
        # TODO: put these arguments to config files
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def draw_faces(self, img, detections):
        for i in range(0, detections.shape[2]):
            (h, w) = img.shape[:2]

            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.config.confidence_threshold:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(img, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return img

    def detect_and_draw_faces(self, img):
        detections = self.detect_faces(img)
        self.draw_faces(img, detections)
