
# Acquiring the necessary library and importing it
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# The names of each character, in the correct order for the Hindi Devnagri script
class_names='''character_1_ka character_2_kha character_3_ga character_4_gha character_5_kna character_6_cha character_7_chha 
character_8_ja character_9_jha character_10_yna character_11_taamatar character_12_thaa character_13_daa character_14_dhaa 
character_15_adna character_16_tabala character_17_tha character_18_da character_19_dha character_20_na character_21_pa 
character_22_pha character_23_ba character_24_bha character_25_ma character_26_yaw character_27_ra character_28_la 
character_29_waw character_30_motosaw character_31_petchiryakha character_32_patalosaw character_33_ha character_34_chhya 
character_35_tra character_36_gya digit_0 digit_1 digit_2 digit_3 digit_4 digit_5 digit_6 digit_7 digit_8 digit_9'''.split()

hindi_character = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ॠ त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'.split()                                                                                                        


# 1
def see_random_picture(target_dir , class_name , display_info = True , return_image_path = False , return_image_array=True):
  '''
  Display a random image from target_dir

  Parameters 
  ----------
  target_dir : Directory from where to pick the random image
  class_name : name of character class/directory
  display_info : Display shape and image (default=True)
  return_image_path : return the random image path (default=False)
  return_image_array : return the image array

  Return
  ---------
  Return the random image in array form 

  '''

  target_directory = target_dir + '/' + class_name                    # Example --> Train + '/' + character_1_ka  -->  Train/character_1_ka
  random_picture_name = random.choice(os.listdir(target_directory))   # Selecting random image from Train/character_1_ka
  image_path = target_directory + '/' + random_picture_name           # let say random image name is 10963.png , so final image path would be :
                                                                      #                                                     Train/character_1_ka/10963.png

  image = plt.imread(image_path) 

  if display_info:

    plt.figure(figsize=(2,2))
    print(f"Shape of image : {image.shape}")  #  display the shape of the image.
    plt.imshow(image , cmap='gray')
    plt.title(class_name)
    plt.axis("off")

  if return_image_path:
    return image_path
  elif return_image_array:
    return image


# 2
from tqdm.notebook import tqdm
def read_image_from_directory(directory , class_names , shuffle = True , verbose = True):
  '''
  directory : This could be either Train or Test directory
  class_names : name of all the character class
  shuffle : Shuffle the dataframe
  verbose : print information if vebose is True

  Return
  -----------
  Return the dataframe, each row is image in 32 by 32 and each column is pixel vale of image and last column is the target class
  '''

  data = []
  for class_name_ in tqdm(class_names):
    if verbose:
      print(f"Converting images of {class_name_} to 1024 dimentional vector")
    target_directory = directory + '/' + class_name_
    random_picture_name = os.listdir(target_directory)

    for name in random_picture_name:           

      image_path = target_directory + '/' + name
      image = cv2.imread(image_path)[:,:,0]            # Read each image 
      image = image.reshape(-1)                        # Reshape each image to 1024 dimention
      image = np.concatenate([image,class_names.index(class_name_)],axis=None)   # concating 1024 dimention of image and the target class
      data.append(image)

  data = np.array(data).astype(int)
  df=pd.DataFrame(data=data)        # Convert the array to DataFrame

  if shuffle:
    df = df.sample(frac=1)

  df.rename(columns={1024:"Target_Class"},inplace=True)   # Change the last column name to target class

  print("DataFrame successfully created !!! ")
  return df

# 3

def display_random_images(data , class_names):
  '''
  Parameters :
  -------------
  data : DataFrame from which random image to be displayed
  class_names : name of all the character class

  Return :
  -------------
  Return random image from provide DataFrame

  '''
  plt.figure(figsize=(10,10))
  for i in range(9):
    plt.subplot(3,3,i+1)

    j = random.randint(0,data.shape[0])

    plt.imshow(data.iloc[j,0:1024].values.reshape(32,32),cmap='gray')
    plt.title(class_names[data.iloc[j,-1]])
    plt.axis(False)
  plt.show()


#4
def find_accuracy_precision_recall_f1(y_true,y_pred):
  '''
  y_true :  Actual labels of images 
  y_pred :  Predicted labels of images

  Return 
  --------
  Return accuracy , precision , recall and f1 Score
  '''
  accuracy = accuracy_score(y_true,y_pred)
  precision , recall , f1 , _ = precision_recall_fscore_support(y_true,y_pred , average='weighted')

  return {
      'accuracy' : accuracy,
      'precision' : precision,
      'recall' : recall,
      'f1' : f1
  }

# 5
def display_wrong_predictions(data,subplots=(10,10),figsize=(10,10),show_probability_score=False,fontsize=15):
  '''
  data : data to be displayed
  subplots : number of plots to create in row and columns wise
  figsize : size of the figure
  show_probability_score : Display the Confidence of predicted class
  fontsize : Size of the title

  '''

  from matplotlib.font_manager import FontProperties
  from pathlib import Path 
  hindi_font = FontProperties(fname=Path('/content/Nirmala.ttf'))

  plt.figure(figsize=figsize)

  for i in range(subplots[0]*subplots[1]):
    plt.subplot(subplots[0],subplots[1],i+1)
    plt.imshow(data.iloc[i,0:1024].values.reshape(32,32),cmap='gray')

    if show_probability_score:
      plt.title(f"Actual Class : {hindi_character[data.iloc[i,1024]]}\nPredicted Class : {hindi_character[data.iloc[i,1025]]}\nConfidence : {data.iloc[i,1026]*100:.2f} %",
                 fontproperties = hindi_font , fontsize = fontsize , color='red')
    else:
      plt.title(f"Actual Class : {hindi_character[data.iloc[i,1024]]}\nPredicted Class : {hindi_character[data.iloc[i,1025]]}",
                 fontproperties = hindi_font , fontsize = fontsize , color='red')
    plt.axis("off")