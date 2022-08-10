# Import required libraries
import os
import random
import shutil
from tqdm.notebook import tqdm

def get_x_percent_images(target_dir , new_dir , percent = 0.1 ):
	'''
	Parameters :
	-----------------
	target_dir = Where the images are to be copied from
	new_dir    = Directory where to copy images
	percent    = What percentage of the data is to be copied (default = 0.1 --> 10 %)
	'''

  # Create list of dictionary with key as directory/class name and values as respected images of that class
  # Ex : [{'character_1_ka':['1340.png','1341.png'.....]},
  #        {'character_2_kha':['1771.png','2772.png'........]},
  #          ...........]
  images = [{dir_name : os.listdir(target_dir + dir_name)} for dir_name in os.listdir(target_dir)]

  for i in images:   # Get one Character class at a time

    for k , v in i.items():   # Get class as key and value as all the images of that class

      ten_percent = round(int(len(v)*percent))    
      random_images = random.sample(v , ten_percent) # Get 10 % random images 

      new_target_dir = new_dir + k  # Create the same class name as present in the target_dir
      os.makedirs(new_target_dir , exist_ok=True)# In the new directory create a class name Ex: NewDirectory/character_1_ka

      for file_name in tqdm(random_images):                  # Copy the randomly selected images to new directory
        original_path = target_dir + k + "/" + file_name
        new_path = new_target_dir + "/" + file_name

        shutil.copy2(original_path,new_path)