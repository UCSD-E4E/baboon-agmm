import numpy as np
import cv2
from scipy.stats import multivariate_normal
import pyjion

pyjion.enable()
pyjion.set_optimization_level(2)

num_gaussians = 3
alpha = 0.01
beta = 0.8
height = None
width = None

cap = cv2.VideoCapture('videos/car-2165.mp4')

ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]

framecount = 0

Mu = np.zeros((height, width, num_gaussians, 3))
Sigma = np.zeros((height, width, num_gaussians))
W = np.zeros((height, width, num_gaussians))
for i in range(height):
    for j in range(width):
        Mu[i,j] = np.array([[122, 122, 122]] * num_gaussians)
        Sigma[i,j] = [36.0] * num_gaussians
        W[i,j] = [1.0/num_gaussians] * num_gaussians
            

def update():
    mask = np.zeros((height, width), dtype=int)
    for i in range(height):
        for j in range (width):
            mask[i, j] = -1
            ratio = []
            for k in range(num_gaussians):
                ratio.append(W[i, j, k]/np.sqrt(Sigma[i, j, k]))
            indices = np.array(np.argsort(ratio[: : -1]))
            Mu[i,j] = Mu[i,j][indices]
            Sigma[i,j] = Sigma[i,j][indices]
            probability = 0
            for l in range(num_gaussians):
                probability += W[i,j,l]
                if probability >= beta and l < num_gaussians - 1:
                    mask[i, j] = l
                    break

            if mask[i, j] == -1:
                mask[i, j] = num_gaussians-2
    return mask


def apply(frame, mask):
    foreground = np.zeros((height, width))
    for i in range(0, frame.shape[0]):
        for j in range(0, frame.shape[1]):
            pixel = frame[i, j]
            match = -1
            for k in range(num_gaussians):
                covarianceInv = np.linalg.inv(Sigma[i,j,k]*np.eye(3))
                pixel_Mu = pixel - Mu[i,j,k]
                dist = np.dot(pixel_Mu.T, np.dot(covarianceInv, pixel_Mu))
                if dist < 6.25*Sigma[i,j,k]:
                    match = k
                    break
            if match != -1:
                W[i,j] = (1.0 - alpha)*W[i,j]
                W[i,j,match] += alpha
                rho = alpha * multivariate_normal.pdf(pixel, Mu[i,j,match], np.linalg.inv(covarianceInv))
                Sigma[match] = (1.0 - rho) * Sigma[i,j,match] + rho * np.dot((pixel - Mu[i,j,match]).T, (pixel - Mu[i,j,match]))
                Mu[i,j,match] = (1.0 - rho) * Mu[i,j,match] + rho * pixel

                if match > mask[i,j]:
                    foreground[i,j] = 250
            else:
                Mu[i,j,-1] = pixel
                foreground[i,j] = 250
    return foreground
                    



while cap.isOpened():
    # read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    framecount += 1 

    # apply AGMM background subtraction to the frame
    model = update()
    background = apply(frame, model)

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