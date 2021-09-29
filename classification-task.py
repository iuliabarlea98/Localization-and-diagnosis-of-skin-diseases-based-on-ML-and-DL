import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import keras
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.metrics import classification_report
plt.style.use('dark_background')
np.random.seed(42)
skin_df = pd.read_csv('D:/ANUL4!!!!!!!!/PNI/HAM10000/incercare/lastaugum.csv')
SIZE=40 
# label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['disease'])
LabelEncoder()
print('The diseases are: ')
print(list(le.classes_))
 
skin_df['label'] = le.transform(skin_df["disease"]) 
print(skin_df.sample(20))



plt.figure(figsize=(20,8))
plt.title('Disease type')
skin_df['disease'].value_counts().plot(kind='pie')
plt.show()
plt.figure(figsize=(20,8))
plt.title('Sex')
skin_df['sex'].value_counts().plot(kind='pie',autopct='%1.1f%%',colors=['#66b3ff','#ff9999','#ffcc99'])
plt.show()
plt.figure(figsize=(20,8))
plt.title('Localization of the disease')
skin_df['localization'].value_counts().plot(kind='bar',color='#ff9999')
plt.show()
plt.figure(figsize=(20,8))
plt.title('Age of the patients')
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='orchid');
plt.title('Age distribution')
plt.show()

# Data distribution visualization
#pie plots
fig = plt.figure(figsize=(20,8))

ax1 = fig.add_subplot(141)
skin_df['disease'].value_counts().plot(kind='bar', ax=ax1,color='purple')
ax1.set_ylabel('Count')
ax1.set_title('Disease Type');

ax2 = fig.add_subplot(142)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2,color='pink')
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(143)
skin_df['localization'].value_counts().plot(kind='bar',color='indigo')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(144)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='orchid');
ax4.set_title('Age')

plt.tight_layout()
plt.show()


print(skin_df['label'].value_counts())


df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]



#read img 
image_path = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('D:/ANUL4!!!!!!!!/PNI/HAM10000/incercare/HERE/', '*', '*.jpg'))}

# path added as a new column
skin_df['path'] = skin_df['image_id'].map(image_path.get)
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))


n_samples = 5  # number of samples for plotting
# Plotting
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['disease']).groupby('disease')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=100).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
        

#Convert dataframe column of images into numpy array
X = np.asarray(skin_df['image'].tolist())
X = X/255.  # Scale values to 0-1
Y=skin_df['label']  #Assign label values to Y
Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical as this is a multiclass classification
#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.25, random_state=50)

#Define the model.
num_classes = 7
model = Sequential()
model.add(Conv2D(256 , (3, 3),activation='relu',padding = 'same', kernel_initializer = 'he_uniform',input_shape=(SIZE, SIZE, 3),kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3),activation='relu',padding = 'same', kernel_initializer = 'he_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3),activation='relu',padding = 'same', kernel_initializer = 'he_uniform',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7, activation='softmax'))
model.summary()
learning_rate=0.0001
optimiser = tf.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['acc'])


# Train

batch_size = 32
epochs = 55

history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size = batch_size,
    validation_data=(x_test, y_test),
    verbose=2)

score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


#plot the training and validation accuracy and loss at each epoch
plt.figure()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Prediction on test data
y_pred = model.predict(x_test)
#predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 
#test data to one hot vectors
y_true = np.argmax(y_test, axis = 1) 

#confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

fig, ax = plt.subplots(figsize=(6,6))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


# fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
plt.figure()
plt.bar(np.arange(7), incorr_fraction)

plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.show()
SAVED_MODEL_PATH="C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/modeldropout02si256ly.h5"
model.save(SAVED_MODEL_PATH)
reconstructed_model = keras.models.load_model("C:/Users/Iulia/OneDrive - Technical University of Cluj-Napoca/Desktop/codes/NEW TRY/modeldropout02si256ly.h5")
reconstructed_model.summary()

score = reconstructed_model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
score1=reconstructed_model.evaluate(x_train,y_train)
print('Train accuracy:',score1[1])
y_pred1=reconstructed_model.predict(x_test)
y_pred_classes1=np.argmax(y_pred1,axis=1)
y_true1=np.argmax(y_test,axis=1)

cm1=confusion_matrix(y_true1,y_pred_classes1)
fig,ax=plt.subplots(figsize=(6,6))
sns.set(font_scale=0.8)
sns.heatmap(cm1,annot=True,linewidths=.5,ax=ax)

incorr_fraction=1-np.diag(cm1)/np.sum(cm1,axis=1)
print(incorr_fraction)
plt.figure()
plt.bar(np.arange(7),incorr_fraction)
plt.xlabel('True label')
plt.ylabel('Fraction of incorrect predictions')
plt.show()

