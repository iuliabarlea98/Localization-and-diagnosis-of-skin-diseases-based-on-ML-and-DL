import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


 
def feature_extraction(img):
    df = pd.DataFrame()
#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
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
                    plt.imshow(fimg.reshape(img.shape))
                    plt.suptitle("Gabor filter with: "+"theta: "+str(theta)+"; sigma: "+str(sigma)+";\n lambda: "+str(lamda))
                    plt.show()
                    #plt.imshow(kernel)
                    #plt.show()
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
########################################
                  
    #CANNY EDGE
    edges = cv2.Canny(img, 0.66*np.mean(img),1.33*np.mean(img))   #Image, min and max values
    edgesinv=np.invert(edges)
    plt.imshow(edgesinv)
    plt.suptitle('Canny')
    plt.show()
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt,meijering
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_robertsinv=np.invert(edge_roberts)
    plt.imshow(edge_robertsinv,cmap='gray')
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
    
    #VARIANCE with size=5
    variance_img = nd.generic_filter(img, np.var, size=5)
    plt.imshow(variance_img,cmap='gray')
    plt.suptitle('Variance with size=5')
    plt.show()
    variance_img1 = variance_img.reshape(-1)
    df['Variance s5'] = variance_img1  #Add column to original dataframe

    df.drop(['Gabor14','Gabor13','Gabor15','Gabor26','Gabor25','Gabor9','Gabor16','Gabor17','Gabor18','Gabor2','Gabor1','Gabor10'], axis = 1,inplace=True)

    return df

 
def feature_extraction1(img):
    df1 = pd.DataFrame()
#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df1['Original Image'] = img2

#Generate Gabor features
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
                    plt.imshow(fimg.reshape(img.shape))
                    plt.suptitle("Gabor filter with: "+"theta: "+str(theta)+"; sigma: "+str(sigma)+";\n lambda: "+str(lamda))
                    plt.show()
                    #plt.imshow(kernel)
                    #plt.show()
                    filtered_img = fimg.reshape(-1)
                    df1[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                    print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                    num += 1  #Increment for gabor column label
########################################
                  
    #CANNY EDGE
    blurred_img = cv2.blur(img,ksize=(5,5))
    med_val = np.median(img) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    edges = cv2.Canny(img, threshold1=lower,threshold2=upper)
    edgesinv1=np.invert(edges)
    plt.imshow(edgesinv1)
    plt.suptitle('Canny')
    plt.show()
    edges1 = edges.reshape(-1)
    df1['Canny Edge'] = edges1 #Add column to original dataframe
    
    from skimage.filters import roberts, sobel, scharr, prewitt,meijering
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_robertsinv1=np.invert(edge_roberts)
    plt.imshow(edge_robertsinv1,cmap='gray')
    plt.suptitle('Roberts')
    plt.show()
    edge_roberts1 = edge_roberts.reshape(-1)
    df1['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    plt.imshow(edge_sobel,cmap='gray')
    plt.suptitle('Sobel')
    plt.show()
    edge_sobel1 = edge_sobel.reshape(-1)
    df1['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    plt.imshow(edge_scharr,cmap='gray')
    plt.suptitle('Scharr')
    plt.show()
    edge_scharr1 = edge_scharr.reshape(-1)
    df1['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    plt.imshow(edge_prewitt,cmap='gray')
    plt.suptitle('Prewitt')
    plt.show()
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df1['Prewitt'] = edge_prewitt1
    
    
    #MEIJERING -Meijering neuriteness filter
    meij = meijering(img,sigmas=(3,5))
    plt.imshow(meij,cmap='gray')
    plt.suptitle('Meijering')
    plt.show()
    meij1 = meij.reshape(-1)
    df1['Meijering'] = meij1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    plt.imshow(gaussian_img,cmap='gray')
    plt.suptitle('Gaussian with sigma=3')
    plt.show()
    gaussian_img1 = gaussian_img.reshape(-1)
    df1['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    plt.imshow(gaussian_img2,cmap='gray')
    plt.suptitle('Gaussian with sigma=7')
    plt.show()
    gaussian_img3 = gaussian_img2.reshape(-1)
    df1['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with size=3
    median_img = nd.median_filter(img, size=5)
    plt.imshow(median_img,cmap='gray')
    plt.suptitle('Median with sigma=5')
    plt.show()
    median_img1 = median_img.reshape(-1)
    df1['Median s3'] = median_img1
    
    #VARIANCE with size=5
    variance_img = nd.generic_filter(img, np.var, size=5)
    plt.imshow(variance_img,cmap='gray')
    plt.suptitle('Variance with size=5')
    plt.show()
    variance_img1 = variance_img.reshape(-1)
    df1['Variance s5'] = variance_img1  #Add column to original dataframe

    df1.drop(['Gabor22','Gabor20','Gabor19','Gabor12','Gabor11','Gabor28','Canny Edge','Gabor4','Gabor3','Gabor27'], axis = 1,inplace=True)

    return df1

import glob
import pickle

filename = "C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/RFmodel_final"
loaded_model = pickle.load(open(filename, 'rb'))


filename1 = "C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/RFCrmodel_final"
loaded_model1 = pickle.load(open(filename1, 'rb'))

path = "D:/PENTRU LICENTA/SEGMENTATION/images/TRAINING/train/*.jpg"
classid=0
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img0= cv2.imread(file)
    img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    #Call the feature extraction function.
    #X = feature_extraction(img)
    #result_texture = loaded_model.predict(X)
    #plt.show()
    
    rgb_image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    shape = rgb_image.shape
    print(shape)
    r, g, b = cv2.split(rgb_image)
 
    ycbcr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
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
    #img=np.concatenate((Y1,Cr))
    
    ###APPLYING ON Cr COMPONENT
    img0=Cr
    Z = feature_extraction1(img0)
    result_texture1 = loaded_model1.predict(Z)
    segmented1 = result_texture1.reshape((img0.shape))
    plt.imshow(segmented1 ,cmap='magma')
    plt.suptitle('Segmentation on Cr component')
    plt.show()
    
    ########APPLYING ON Y COMPONENT
    img1=Y
    K = feature_extraction(img1)
    result_texture2 = loaded_model.predict(K)
    segmented2 = result_texture2.reshape((img1.shape))
    plt.imshow(segmented2)
    plt.suptitle('Segmentation on Y component')
    plt.show()
    
    
    #frames=[Z,K]
    #concatenation=pd.concat(frames)
    #print(concatenation.head()) 
    #result_texture3=loaded_model.predict(concatenation)
    #segmented3=result_texture3.reshape((img1.shape))
    #plt.imshow(segmented3)
    #plt.suptitle('Segmentation on the concatenation of Y and Cb')
    #plt.show()
   
    
    #plt.imsave('C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/texturaa/'+str(classid)+'segmvar2img.jpg', segmented1, cmap ='jet')
    #plt.imsave('C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/culoaree/'+str(classid)+'segmvar2img.jpg', segmented2, cmap ='inferno')
    classid+=1





