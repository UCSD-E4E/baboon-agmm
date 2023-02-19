import cv2

class AGMM:
    def __init__(self, alpha=0.05, k=3, std_dev=2.5, prune_threshold=0.05):
        self.alpha = alpha
        self.k = k
        self.std_dev = std_dev
        self.prune_threshold = prune_threshold
        self.background = None
        self.variances = None
        self.weights = None


# Usage example
cap = cv2.VideoCapture('videos/car-2165.mp4')
agmm = AGMM()

while cap.isOpened():
    # read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # apply AGMM background subtraction to the frame

    # display the foreground mask
    cv2.imshow('FG Mask', frame)

    # check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()