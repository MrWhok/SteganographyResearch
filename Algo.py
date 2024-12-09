import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim

class Steganography:
    def embed(self, cover_image_path, payload_path):
        # Open the cover image
        X = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        w, h = X.shape
        X_flatten = X.flatten()

        KEY = len(X_flatten) * [None]

        cp_x1 = np.copy(X_flatten)
        cp_x2 = np.copy(X_flatten)

        # Open the payload
        with open(payload_path, 'r') as file:
            payload_data = file.read().strip().split('\n')
        Payload = np.array([int(char) for char in payload_data])

        # Checking pixel that can't be assigned the payload
        KEY, x1, x2 = self._iterate_image(X_flatten, Payload, KEY, cp_x1, cp_x2)

        X1 = x1.reshape(w, h)
        X2 = x2.reshape(w, h)

        psnr_value1 = cv2.PSNR(X, X1)
        psnr_value2 = cv2.PSNR(X, X2)
        print("PSNR Value1:", psnr_value1)
        print("PSNR Value2:", psnr_value2)

        mse_value1 = self.calculate_mse(X, X1)
        mse_value2 = self.calculate_mse(X, X2)
        print("MSE Value1:", mse_value1)
        print("MSE Value2:", mse_value2)

        ssim_value1 = ssim(X, X1)
        ssim_value2 = ssim(X, X2)
        print("SSIM Value1:", ssim_value1)
        print("SSIM Value2:", ssim_value2)

        return X1, X2, KEY, Payload

    def extract(self, X1, X2, KEY):
        X1_flatten = X1.flatten()
        X2_flatten = X2.flatten()
        Payload = []
        for i in range(len(X1_flatten)):
            # if KEY[i] == 0:
            #     Payload.append(0)
            if KEY[i] == 2:
                Payload.append(abs(X1_flatten[i] - X2_flatten[i]))
            elif KEY[i] == 1:
                Payload.append(abs(X1_flatten[i] - X2_flatten[i]) + 1)
        return Payload

    def _iterate_image(self, X_flatten, Payload, KEY, cp_x1, cp_x2):
        payload_index = 0  # Initialize payload index
        for i in range(len(X_flatten)):
            if payload_index < len(Payload):
                if X_flatten[i] < 4 or X_flatten[i] > 251:
                    KEY[i] = 3
                    cp_x1[i] = X_flatten[i]
                    cp_x2[i] = X_flatten[i]
                else:
                    Temp = Payload[payload_index] // 2
                    if Payload[payload_index] % 2 == 0:  # If the payload is even
                        KEY[i] = 2
                        cp_x1[i] = X_flatten[i] + Temp
                        cp_x2[i] = X_flatten[i] - Temp
                    elif Payload[payload_index] % 2 == 1:  # If the payload is odd
                        KEY[i] = 1
                        cp_x1[i] = X_flatten[i] + Temp
                        cp_x2[i] = X_flatten[i] - Temp
                    payload_index += 1  # Increment payload index
            else:
                KEY[i] = 3
                cp_x1[i] = X_flatten[i]
                cp_x2[i] = X_flatten[i]
        return KEY, cp_x1, cp_x2

    def calculate_mse(self, imageA, imageB):
        # Mean Squared Error (MSE) between two images
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        return err

stego = Steganography()

cover_image_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/Images/15.tiff'
payload_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/Payload/random_numbers100.txt'

X1, X2, KEY, payloadOri = stego.embed(cover_image_path, payload_path)

# cv2.imshow('Image with Payload 1', X1)
# cv2.imshow('Image with Payload 2', X2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# EXTRACT
payloadExtract = stego.extract(X1, X2, KEY)

if np.array_equal(payloadOri, payloadExtract):
    print("Payload is same")
else:
    print("Payload is different")
