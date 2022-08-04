
# Acquiring the necessary library and importing it
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

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

  from pathlib import Path 
  from matplotlib.font_manager import FontProperties

  # configure the Hindi font
  if not os.path.exists("Nirmala.ttf"):

    print("Downloading the Nirmala.tff ...")
    !wget https://www.wfonts.com/download/data/2016/04/29/nirmala-ui/nirmala-ui.zip
    !unzip -q nirmala-ui.zip

  else:   
    print("Nirmala.ttf already exist. Skipping downloading...")
  
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