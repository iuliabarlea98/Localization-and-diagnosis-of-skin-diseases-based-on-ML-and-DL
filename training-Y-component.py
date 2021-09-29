import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
plt.style.use('dark_background') 

image_dataset = pd.DataFrame()  #Dataframe to capture image features

img_path = "D:/PENTRU LICENTA/SEGMENTATION/images/TRAINING/train/"
for image in os.listdir(img_path):  #iterate through each file 
    print(image)
    
    df = pd.DataFrame() 
    input_img = cv2.imread(img_path + image)  #Read images
    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
        ##YCbCr
        ##img2=Cr
        img2=img.reshape(-1)
        df['Pixel_Value'] = img2
        df['Image_Name'] = image 
        
    # Gabor features(32)
    num = 1 
    kernels = []
    for theta in (0,1):
        theta = (theta / 4 )* np.pi
        for sigma in (1, 3):  
            for lamda in np.arange(0, np.pi, np.pi / 4):  
                for gamma in (0.05, 0.5):  
                    gabor_label = 'Gabor' + str(num) 
                    psi=0
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    #plt.imshow(kernel)
                    #plt.show()
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
    ########################################
      
    #CANNY EDGE
    edges = cv2.Canny(img, 0.66*np.mean(img),1.33*np.mean(img))   #Image, min and max values
    plt.imshow(edges,cmap='gray')
    plt.suptitle('Canny')
    plt.show()
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt,meijering
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    plt.imshow(edge_roberts,cmap='gray')
    plt.suptitle('Roberts')
    plt.show()
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    plt.imshow(edge_sobel,cmap='gray')
    plt.suptitle('Sobel')
    plt.show()
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    plt.imshow(edge_scharr,cmap='gray')
    plt.suptitle('Scharr')
    plt.show()
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    plt.imshow(edge_prewitt,cmap='gray')
    plt.suptitle('Prewitt')
    plt.show()
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    
    #MEIJERING -Meijering neuriteness filter
    meij = meijering(img,sigmas=(3,5))
    plt.imshow(meij,cmap='gray')
    plt.suptitle('Meijering')
    plt.show()
    meij1 = meij.reshape(-1)
    df['Meijering'] = meij1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    plt.imshow(gaussian_img,cmap='gray')
    plt.suptitle('Gaussian with sigma=3')
    plt.show()
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    plt.imshow(gaussian_img2,cmap='gray')
    plt.suptitle('Gaussian with sigma=7')
    plt.show()
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with size=3
    median_img = nd.median_filter(img, size=5)
    plt.imshow(median_img,cmap='gray')
    plt.suptitle('Median with sigma=5')
    plt.show()
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=5)
    plt.imshow(variance_img,cmap='gray')
    plt.suptitle('Variance with size=5')
    plt.show()
    variance_img1 = variance_img.reshape(-1)
    df['Variance s5'] = variance_img1  #Add column to original dataframe
    
    
    df.drop(['Gabor14','Gabor13','Gabor15','Gabor26','Gabor25','Gabor9','Gabor16','Gabor17','Gabor18','Gabor2','Gabor1','Gabor10'], axis = 1,inplace=True)
    image_dataset = image_dataset.append(df)

    
print(df.head())
    
mask_dataset = pd.DataFrame()  #Create dataframe to capture mask info.

mask_path = "D:/PENTRU LICENTA/SEGMENTATION/images/apeer/masks/"    
for mask in os.listdir(mask_path):
    print(mask)
    
    df2 = pd.DataFrame() 
    input_mask = cv2.imread(mask_path + mask)
    label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    labeled_img2 = label.reshape(-1)
    df2['Labels'] = labeled_img2
    df2['Mask_Name'] = mask 
    mask_dataset = mask_dataset.append(df2)

dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets
dataset = dataset[dataset.Labels != 0]

X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Labels"], axis=1) 

#Assign label values to Y (prediction)
Y = dataset["Labels"].values 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=20)

#Import training classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

##Save the trained model as pickle 
model_name = "C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/RFmodel_final"
pickle.dump(model, open(model_name, 'wb'))

filename = "C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/RFmodel_final"
loaded_model = pickle.load(open(filename, 'rb'))
accc=loaded_model.predict(X_test)
accc1=loaded_model.predict(X_train)
print("Accuracy on test set: ",metrics.accuracy_score(y_test,accc))
print("Accuracy on training set: ",metrics.accuracy_score(y_train,accc1))
