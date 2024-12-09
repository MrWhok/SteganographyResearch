import cv2
import math
import numpy as np

class Steganography:
    def embed(self, cover_image_path, payload_path):
        #Open the cover image
        X = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        w, h = X.shape
        X_flatten = X.flatten()

        KEY=len(X_flatten)*[None]

        cp_x1 = np.copy(X_flatten)
        cp_x2 = np.copy(X_flatten)

        #Open the payload
        with open(payload_path, 'r') as file:
            payload_data = file.read().strip().split('\n')
        Payload = np.array([int(char) for char in payload_data])

        #Checking pixel that cant assigned the payload
        KEY,x1,x2=self._iterate_image(X_flatten,Payload,KEY,cp_x1,cp_x2)

        X1 = x1.reshape(w, h)
        X2 = x2.reshape(w, h)

        psnr_value1 = cv2.PSNR(X, X1)
        psnr_value2 = cv2.PSNR(X, X2)
        print("PSNR Value1:", psnr_value1)
        print("PSNR Value2:", psnr_value2)
        return X1, X2, KEY, Payload

    def extract(self,X1,X2,KEY):
        X1_flatten=X1.flatten()
        X2_flatten=X2.flatten()
        Payload=[]
        for i in range(len(X1_flatten)):
            if KEY[i]==0:
                Payload.append(0)
            elif KEY[i]==1:
                Payload.append(abs(X1_flatten[i]-X2_flatten[i]))
            elif KEY[i]==2:
                Payload.append(abs(X1_flatten[i]-X2_flatten[i])+1)
        return Payload

    def _iterate_image(self, X_flatten,Payload, KEY,cp_x1,cp_x2):
        counter=len(Payload)
        for i in range(len(X_flatten)):
            if(counter):
                if X_flatten[i]<4 or X_flatten[i]>251:
                    KEY[i]=3
                    cp_x1[i]=X_flatten[i]
                    cp_x2[i]=X_flatten[i]
                else:
                    if(counter):
                        Temp=Payload[i]//2
                        if Payload[i]==0: #If the payload is 0
                            KEY[i]=0
                            cp_x1[i]=X_flatten[i]
                            cp_x2[i]=X_flatten[i]
                        elif Payload[i]%2==0: #If the payload is even
                            KEY[i]=1
                            cp_x1[i]=X_flatten[i]+Temp
                            cp_x2[i]=X_flatten[i]-Temp
                        elif Payload[i]%2==1: #If the payload is odd
                            KEY[i]=2
                            cp_x1[i]=X_flatten[i]+Temp
                            cp_x2[i]=X_flatten[i]-Temp
                        counter-=1
            else:
                KEY[i]=3
                cp_x1[i]=X_flatten[i]
                cp_x2[i]=X_flatten[i]
        return KEY,cp_x1,cp_x2

stego=Steganography()

cover_image_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/7.1.03.tiff'
payload_path = '/home/aydin/Vscode/Kuliah/SteganographyResearch-master/dualImage/random_numbers.txt'

X1,X2,KEY,payloadOri=stego.embed(cover_image_path, payload_path)

# cv2.imshow('Image with Payload 1', X1)
# cv2.imshow('Image with Payload 2', X2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#EXTRACT
payloadExtract=stego.extract(X1,X2,KEY)

if np.array_equal(payloadOri, payloadExtract):
    print("Payload is same")
else:
    print("Payload is different")
