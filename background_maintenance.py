import numpy as np
from scipy.stats import norm
import cv2

class AGMM:
    def __init__(self, N=3, alpha=0.05, T_sigma=2.5, sigma_0=100.0, w_0=0.01):
        self.N = N
        self.alpha = alpha
        self.T_sigma = T_sigma
        self.sigma_0 = sigma_0
        self.w_0 = w_0
        self.mu = np.zeros((N,))
        self.sigma = np.zeros((N,))
        self.w = np.zeros((N,))
        
    def update(self, frame):
        height, width = frame.shape[:2]
        model_match = np.zeros((height, width, self.N), dtype=bool)
        distance = np.zeros((height, width, self.N)) + np.inf
        
        # Model matching
        for n in range(self.N):
            dist = np.abs(frame - self.mu[n])
            condition = dist <= self.T_sigma * self.sigma[n]
            distance[condition, n] = -self.w[n]
        
        argmin = np.argmin(distance, axis=2)
        argmin[distance[argmin, np.arange(argmin.shape[1])] == np.inf] = -1
        np.putmask(model_match, distance != np.inf, True)
        # Model renewing
        self.w = (1 - self.alpha) * self.w + self.alpha * model_match
        replace_mask = (np.sum(model_match, axis=2) == 0)
        if np.any(replace_mask):
            k = np.argmin(self.w)
            self.mu[k] = frame[replace_mask]
            self.sigma[k] = self.sigma_0
            self.w[k] = self.w_0

        update_mask = ~replace_mask
        if np.any(update_mask):
            alpha_t = self.alpha / np.sum(model_match[update_mask], axis=2)
            alpha_t = alpha_t.reshape(-1, self.N)
            alpha_t[~np.isfinite(alpha_t)] = 0
            p_t = alpha_t * norm.pdf(frame[update_mask, None], self.mu, self.sigma)
            p_t[~np.isfinite(p_t)] = 0
            
            # Update phase
            self.mu = (1 - p_t) * self.mu + p_t * frame[update_mask, None]
            self.sigma = (1 - p_t) * self.sigma + p_t * np.square(frame[update_mask, None] - self.mu)
        
        self.w = self.w / np.sum(self.w)

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