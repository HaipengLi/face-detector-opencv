class CascadeFaceDetectorConfig:
    classifier_path = 'cascades/haarcascade_frontalface_default.xml'
    scale_factor = 1.2
    min_neighbors = 5
    min_size = (20, 20)


class OpencvFaceDetectorConfig:
    prototxt_path = 'models/deploy.prototxt.txt'
    model_weights_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    confidence_threshold = 0.5

