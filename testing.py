import cv2
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
import openpyxl

class Steganography:
    def embed(self, cover_image_path, payload_path):
        # Open the cover image
        X = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        if X is None:
            print(f"Error: Unable to open image file {cover_image_path}")
            return None, None, None, None, None, None, None
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

        return X1, X2, KEY, Payload, psnr_value1, psnr_value2, mse_value1, mse_value2, ssim_value1, ssim_value2

    def extract(self, X1, X2, KEY):
        X1_flatten = X1.flatten()
        X2_flatten = X2.flatten()
        Payload = []
        for i in range(len(X1_flatten)):
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

# Initialize Steganography object
stego = Steganography()

# Create a new Excel workbook and select the active worksheet
workbook = openpyxl.Workbook()
sheet = workbook.active

# Write headers to the worksheet
headers = ["Image", "Payload", "PSNR Value 1", "PSNR Value 2", "MSE Value 1", "MSE Value 2", "SSIM Value 1", "SSIM Value 2"]
sheet.append(headers)

# Loop through images and payloads and fill the spreadsheet with corresponding output
for i in range(1, 11):
    for j in range(10, 101, 10):
        cover_image_path = f'/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/Images/{i}.tiff'
        payload_path = f'/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/Payload/random_numbers{j}.txt'
        
        print(f"Processing Image: {cover_image_path}, Payload: {payload_path}")
        
        X1, X2, KEY, payloadOri, psnr_value1, psnr_value2, mse_value1, mse_value2, ssim_value1, ssim_value2 = stego.embed(cover_image_path, payload_path)
        
        if psnr_value1 is not None:
            # EXTRACT
            payloadExtract = stego.extract(X1, X2, KEY)

            if not np.array_equal(payloadOri[:len(payloadExtract)], payloadExtract):
                print(f"Payload mismatch for Image: {cover_image_path}, Payload: {payload_path}")
                print("Original Payload:", payloadOri[:100])  # Print first 100 original payload values for debugging
                print("Extracted Payload:", payloadExtract[:100])  # Print first 100 extracted payload values for debugging
                raise ValueError("Payload mismatch detected. Stopping execution.")
            
            # Append results to the worksheet
            sheet.append([f'{i}.tiff', f'random_numbers{j}.txt', psnr_value1, psnr_value2, mse_value1, mse_value2, ssim_value1, ssim_value2])

# Save the workbook to a file
workbook.save('/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/result2.xlsx')

print("Spreadsheet has been filled with corresponding output.")
