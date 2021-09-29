import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
plt.style.use('dark_background') 

image_dataset = pd.DataFrame()  #Dataframe to capture image features

img_path = "D:/PENTRU LICENTA/SEGMENTATION/images/TRAINING/train/"
for image in os.listdir(img_path):  #iterate through each file 
    print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    input_img = cv2.imread(img_path + image)  #Read images
    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
        ycbcr_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(ycbcr_image)
        plt.imshow(Y,plt.cm.gray)
        plt.suptitle('Y-luminance data')
        plt.show()
        plt.imshow(Cr,plt.cm.gray)
        plt.suptitle('Cr-Chrominance-red channel')
        plt.show()
        plt.imshow(Cb,plt.cm.gray)
        plt.suptitle('Cb-Chrominance-blue channel')
        plt.show()
        ##YCbCr
        img=Cr
        img2=img.reshape(-1)
        df['Pixel_Value'] = img2
        df['Image_Name'] = image 
        
    # Gabor features(32)
    num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
    kernels = []
    for theta in (0,1):   #Define number of thetas
        theta = (theta / 4 )* np.pi
        for sigma in (1, 3):  #Sigma with 1 and 5
            for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
                for gamma in (0.05, 0.5):   #Gamma values of 0.05 and 0.5
                    gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                    psi=0
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter the image and add values to a new column 
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    #plt.imshow(kernel)
                    #plt.show()
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
                    
    ########################################
    #Gerate OTHER FEATURES and add them to the data frame
                    
    #CANNY EDGE
    
    blurred_img = cv2.blur(img,ksize=(5,5))
    med_val = np.median(img) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    edges = cv2.Canny(img, threshold1=lower,threshold2=upper)
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
    
    df.drop(['Gabor22','Gabor20','Gabor19','Gabor12','Gabor11','Gabor28','Canny Edge','Gabor4','Gabor3','Gabor27'], axis = 1,inplace=True)
    
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

dataset.to_csv("D:/PENTRU LICENTA/pt textura/GaborCr.csv")

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
#First test prediction on the training data itself. Should be good. 
prediction_test_train = model.predict(X_train)
print(prediction_test_train)
prediction_test = model.predict(X_test)

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
print ("Accuracy = ", metrics.accuracy_score(y_train,prediction_test_train ))


importances = list(model.feature_importances_)
#Let us print them into a nice format.
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)






##Save the trained model as pickle string to disk for future use
model_name = "C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/RFCrmodel_final"
pickle.dump(model, open(model_name, 'wb'))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


print(confusion_matrix(y_test,prediction_test))
print(classification_report(y_test,prediction_test))
print(accuracy_score(y_test, prediction_test))
y_true1=np.argmax(y_test)
y_pred_classes1=np.argmax(prediction_test)
cm1=confusion_matrix(y_true1,y_pred_classes1)
fig,ax=plt.subplots(figsize=(6,6))
sns.set(font_scale=0.8)
sns.heatmap(cm1,annot=True,linewidths=.5,ax=ax)

    