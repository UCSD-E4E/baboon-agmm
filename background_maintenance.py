import numpy as np
from scipy.stats import norm
import cv2

class AGMM:
    def __init__(self, num_gaussians=3, alpha=0.01, T_sigma=2.5, Sigma_0=1e2, w_0=0.01):
        self.num_gaussians = num_gaussians
        self.alpha = alpha
        self.T_sigma = T_sigma
        self.Sigma_0 = Sigma_0
        self.w_0 = w_0
        
        # Initialize variables
        self.Mu = np.zeros((num_gaussians, 3))
        self.Sigma = np.zeros((num_gaussians, 3))
        self.W = np.zeros(num_gaussians)
        self.frame_count = 0


    def apply(self, frame):
        # Increment frame count
        self.frame_count += 1
        
        # Convert frame to float
        frame = frame.astype(float)

        # Create binary mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Loop over all pixels
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                pixel = frame[i, j, :]

                # Check if pixel matches a model
                match = False
                for n in range(self.num_gaussians):
                    if self.W[n] > 0:
                        if abs(pixel - self.Mu[n]).sum() <= self.T_sigma * np.sqrt(self.Sigma[n]).all():
                            match = True
                            self.W[n] = (1 - self.alpha) * self.W[n] + self.alpha
                            self.Mu[n] = (1 - self.alpha) * self.Mu[n] + self.alpha * pixel
                            self.Sigma[n] = (1 - self.alpha) * self.Sigma[n] + self.alpha * (pixel - self.Mu[n])**2
                            break
                
                # If no match is found, replace the weakest model
                if not match:
                    k = np.argmin(self.W)
                    self.W[k] = self.w_0
                    self.Mu[k] = pixel
                    self.Sigma[k] = self.Sigma_0
                
                # Normalize weights
                self.W /= np.sum(self.W)

                # Create mask
                if np.max(self.W) >= 0.5:
                    mask[i, j] = 255

        # Return binary mask
        return mask
        


# Usage example
cap = cv2.VideoCapture('videos/car-2165.mp4')
agmm = AGMM()

while cap.isOpened():
    # read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # apply AGMM background subtraction to the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = agmm.apply(frame)

    # display the foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('background', background)

    # check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()