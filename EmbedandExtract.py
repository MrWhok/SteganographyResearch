import cv2
import math
import numpy as np

class Steganography:
    def __init__(self, n):
        self.n = n
        self.k = n + 1

    def embed(self, cover_image_path, payload_path):
        X = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        w, h = X.shape
        X_flatten = X.flatten()

        d = self._get_d(X_flatten)
        cp_x1 = np.copy(X_flatten)
        cp_x2 = np.copy(X_flatten)

        with open(payload_path, 'r') as file:
            payload_data = file.read().strip().split('\n')
        Payload = np.array([int(char) for char in payload_data])

        new_payload, length_base10_conv, padding = self._secret_data_step(Payload)
        d1, d2 = self._center_shift_operation(new_payload, d)
        x1, x2 = self._get_stego(cp_x1, cp_x2, d1, d2)

        X1 = x1.reshape(w, h)
        X2 = x2.reshape(w, h)

        psnr_value1 = cv2.PSNR(X, X1)
        psnr_value2 = cv2.PSNR(X, X2)
        print("PSNR Value1:", psnr_value1)
        print("PSNR Value2:", psnr_value2)

        return X1, X2, Payload, length_base10_conv, padding

    def extract(self, stego_image1, stego_image2, original_shape, payload_length, padding):
        w, h = original_shape
        x1 = stego_image1.flatten()
        x2 = stego_image2.flatten()

        d_aks2 = self._calculate_difference(x1, x2)
        X_recovered = self._get_cover_image(x1, x2)
        d_recovered = self._get_d(X_recovered)
        s = self._calculate_secret_data_base10(d_recovered, d_aks2)
        S10 = s[:payload_length]

        S2 = self._convert_to_base2_with_n_digits(S10, self.k, padding)
        concatenated_base2 = self._convert_and_concatenate(S2)
        secret_data = self._convert_to_base10_per_8digits(concatenated_base2)

        return secret_data

    def _secret_data_step(self, payload):
        binary_payload = np.array([format(num, '08b') for num in payload])
        concatenated_payload = ''.join(binary_payload)
        padding = self.k - len(concatenated_payload) % self.k

        sliced_payload = [concatenated_payload[i:i+self.k] for i in range(0, len(concatenated_payload), self.k)]
        base10_payload = np.array([int(binary_string, 2) for binary_string in sliced_payload])
        length_base10_conv = len(base10_payload)
        return base10_payload, length_base10_conv, padding

    def _center_shift_operation(self, s, d):
        d_aks1 = self._get_d_aks1(s, d)
        d_aks2 = self._get_d_aks2(d_aks1)
        d1 = self._calculate_d1(d_aks2)
        d2 = self._calculate_d2(d_aks2)
        return d1, d2

    def _get_stego(self, cp_x1, cp_x2, d1, d2):
        x1 = self._calculate_stego(cp_x1, d1)
        x2 = self._calculate_stego(cp_x2, d2)
        return x1, x2

    def _get_d(self, x):
        denom = pow(2, 8 - self.n)
        d = [math.floor(int(x[i]) / denom) for i in range(len(x))]
        return np.array(d, dtype=int)

    def _get_d_aks1(self, s, d):
        return [s[i] - d[i] for i in range(len(s))]

    def _get_d_aks2(self, d_aks1):
        subtractor = pow(2, self.n - 1)
        return [d_aks1[i] - subtractor for i in range(len(d_aks1))]

    def _calculate_d1(self, d_aks2):
        return [math.floor(d_aks2[i] / 2) for i in range(len(d_aks2))]

    def _calculate_d2(self, d_aks2):
        return [math.floor(d_aks2[i] / (-2)) for i in range(len(d_aks2))]

    def _calculate_stego(self, cp_x, d):
        stego = cp_x.copy()
        for i in range(len(cp_x)):
            if i < len(d):
                stego[i] = cp_x[i] + d[i]
        return stego

    def _calculate_difference(self, x1, x2):
        x1_signed = x1.astype(np.int16)
        x2_signed = x2.astype(np.int16)
        return x1_signed - x2_signed

    def _get_cover_image(self, x1, x2):
        return np.array([math.ceil((int(x1[i]) + int(x2[i])) / 2) for i in range(len(x1))])

    def _calculate_secret_data_base10(self, d, d_aks2):
        temp = pow(2, self.n - 1)
        return [d[i] + d_aks2[i] + temp for i in range(len(d))]

    def _convert_to_base2_with_n_digits(self, array, n, padding):
        base2_array = [format(num, f'0{n}b') for num in array]
        if padding > 0 and len(base2_array) > 0:
            base2_array[-1] = base2_array[-1][-n + padding:]
        return base2_array

    def _convert_and_concatenate(self, array):
        return ''.join(array)

    def _convert_to_base10_per_8digits(self, binary_string):
        chunks = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
        return [int(chunk, 2) for chunk in chunks]

# Example usage
n = 1
stego = Steganography(n)

# Embedding
cover_image_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/7.1.03.tiff'
payload_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/random_numbers.txt'
X1, X2, Payload, length_base10_conv, padding = stego.embed(cover_image_path, payload_path)

# Extracting
original_shape = (512, 512)  # Example shape, replace with actual
secret_data = stego.extract(X1, X2, original_shape, length_base10_conv, padding)

if np.array_equal(secret_data, Payload):
    print("Data is same")
else:
    print("Data is different")
