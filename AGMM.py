import numpy as np
import cython
import cv2
from scipy.stats import multivariate_normal

@cython.cclass
class AGMM():
    num_gaussians: cython.int
    alpha: cython.float
    beta: cython.float
    height: cython.int
    width: cython.int

    def __init__(self, width=640, height=360):
        self.num_gaussians = 3
        self.alpha = 0.01
        self.beta = 0.8
        self.height = height
        self.width = width

        # Initialize variables
        self.Mu = np.zeros((self.height, self.width, self.num_gaussians, 3))
        self.Sigma = np.zeros((self.height, self.width, self.num_gaussians))
        self.W = np.zeros((self.height, self.width, self.num_gaussians))
        i: cython.int
        j: cython.int
        for i in range(self.height):
            for j in range(self.width):
                self.Mu[i,j] = np.array([[122, 122, 122]] * self.num_gaussians)
                self.Sigma[i,j] = [36.0] * self.num_gaussians
                self.W[i,j] = [1.0/self.num_gaussians] * self.num_gaussians
        

    def update(self):
        mask = np.zeros((self.height, self.width), dtype=int)
        i: cython.int
        j: cython.int
        k: cython.int
        l: cython.int
        probability: cython.float
        for i in range(self.height):
            for j in range (self.width):
                mask[i, j] = -1
                ratio = []
                for k in range(self.num_gaussians):
                    ratio.append(self.W[i, j, k]/np.sqrt(self.Sigma[i, j, k]))
                indices = np.array(np.argsort(ratio[: : -1]))
                self.Mu[i,j] = self.Mu[i,j][indices]
                self.Sigma[i,j] = self.Sigma[i,j][indices]
                probability = 0
                for l in range(self.num_gaussians):
                    probability += self.W[i,j,l]
                    if probability >= self.beta and l < self.num_gaussians - 1:
                        mask[i, j] = l
                        break

                if mask[i, j] == -1:
                    mask[i, j] = self.num_gaussians-2
        return mask
    
    def apply(self, frame, mask):
        foreground = np.zeros((self.height, self.width))
        i: cython.int
        j: cython.int
        k: cython.int
        match: cython.float
        for i in range(self.height):
            for j in range(self.width):
                pixel = frame[i, j]
                match = -1
                for k in range(self.num_gaussians):
                    covarianceInv = np.linalg.inv(self.Sigma[i,j,k]*np.eye(3))
                    pixel_Mu = pixel - self.Mu[i,j,k]
                    dist = np.dot(pixel_Mu.T, np.dot(covarianceInv, pixel_Mu))
                    if dist < 6.25*self.Sigma[i,j,k]:
                        match = k
                        break
                if match != -1:
                    self.W[i,j] = (1.0 - self.alpha)*self.W[i,j]
                    self.W[i,j,match] += self.alpha
                    rho = self.alpha * multivariate_normal.pdf(pixel, self.Mu[i,j,match], np.linalg.inv(covarianceInv))
                    self.Sigma[match] = (1.0 - rho) * self.Sigma[i,j,match] + rho * np.dot((pixel - self.Mu[i,j,match]).T, (pixel - self.Mu[i,j,match]))
                    self.Mu[i,j,match] = (1.0 - rho) * self.Mu[i,j,match] + rho * pixel

                    if match > mask[i,j]:
                        foreground[i,j] = 250
                else:
                    self.Mu[i,j,-1] = pixel
                    foreground[i,j] = 250
        return foreground
                    

cap = cv2.VideoCapture('videos/car-2165.mp4')
agmm = AGMM()
framecount = 0

while cap.isOpened():
    # read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    framecount += 1 

    # apply AGMM background subtraction to the frame
    model = agmm.update()
    background = agmm.apply(frame, model)

    # display the foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('background', background)

    if (framecount % 10 == 1):
        cv2.imwrite("render{0}.png".format(framecount), background)

    # check for user input to exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()