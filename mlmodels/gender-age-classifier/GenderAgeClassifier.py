import time
import numpy as np
import urllib.request
import os
import cv2
import pandas as pd
import face_recognition
from shutil import rmtree
from pyagender import PyAgender
from PIL import Image


class GenderAgeClassifier(object):

    def __init__(self):
        #Change the cache_dir parameter to control where profile images are downloaded
        self.path = os.getcwd()
        self.cache_dir = os.path.join(self.path,'twitter_cache')

    def predict(self, data):
        """
        Process

        Parameters
        ----------
        test_data : 'pandas.core.frame.DataFrame'

        Returns
        -------
        result : 1D numpy.array (dtype=int64)
            Prediction result. 0 - female, 1 - male, -1 - unknown
        """
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        gender_list = []
        age_list = []
        count = 0
        invalid_url = 0
        for index, row in data.iterrows():
            try:
                gender = -1
                age = '-1'
                count += 1
                # extract key
                id_str = str(row['tweet_id'])
                img_url = row['profile_image_url_https']
                default_img = str(row['default_profile_image'])

                if (default_img == '0'):
                    # download profile image
                    img_name = '{}.{}'.format(id_str, self.get_extension(img_url))
                    img_file = os.path.join(self.cache_dir, img_name)
                    urllib.request.urlretrieve(img_url, img_file)
                    if len(os.listdir(self.cache_dir)) > 0:
                        # resize image for better face recognition
                        image = Image.open(img_file)
                        new = image.resize((224,224))
                        new.save(img_file)
                        # face recognition
                        image = face_recognition.load_image_file(img_file)
                        face_locations = face_recognition.face_locations(image)
                        if len(face_locations) == 1:
                            gender, age = self.predict_gender_age(img_file)

                    if os.path.exists(img_file):
                        os.remove(img_file)

            except urllib.error.HTTPError:
                invalid_url += 1
            except FileNotFoundError:
                invalid_url += 1

            gender_list.append(gender)
            age_list.append(age)

        gender_arr = np.array(gender_list)
        age_arr = np.array(age_list)
        return gender_arr, age_arr


    def predict_gender_age(self, img):
        """
        Predict function. Using aristofun/py-agender package.
        Parameters
        ----------
        img: str
            Image path
        
        Returns
        -------
        gender : int
            Gender prediction result. 0 - female, 1 - male, -1 - unknown
        
        age: str
            Age prediction result. '-1' - unknown
        """
        gender = -1
        age = '-1'
        agender = PyAgender()
        faces = agender.detect_genders_ages(cv2.imread(img))
        if (len(faces) != 1):
            return gender, age  #"unknown"

        # extract gender prediction
        prob = faces[0]['gender']
        if prob > 0.5:
            gender = 0  #"female"
        else:
            gender = 1  #"male"

        # extract age prediction
        age = faces[0]['age']
        age = self.divide_age(age)
        return gender, age


    def divide_age(self, age):
        """
        Get corresponding age group.

        Parameters
        ----------
        age : int
            Predicted age.

        Returns
        -------
        age : str
            Corresponding age group.
        """
        age_list = ['<20', '20-30', '30-40', '40-50', '50-60', '>60']
        if age < 20:
            return age_list[0]
        elif 20 <= age < 30:
            return age_list[1]
        elif 30 <= age < 40:
            return age_list[2]
        elif 40 <= age < 50:
            return age_list[3]
        elif 50 <= age < 60:
            return age_list[4]
        else:
            return age_list[5]

    def get_extension(self, img_path):
        """
        Get image extension from profile image url.

        Parameters
        ----------
        img: str
            Profile image url

        Returns
        -------
        extension : str
            Image extension
        """
        dotpos = img_path.rfind(".")
        extension = img_path[dotpos + 1:]
        if extension.lower() == "gif":
            return "png"
        return extension


if __name__ == '__main__':
    GAC = GenderAgeClassifier()
    test1_id = 815719251
    test1_img = 'https://pbs.twimg.com/profile_images/645403816213393408/cb9Ur4jH_normal.jpg'
    test2_id = 815719280
    test2_img = 'https://pbs.twimg.com/profile_images/589543258516279296/5ZOZ6uT__normal.jpg'
    test3_id = 529422070
    test3_img = 'https://pbs.twimg.com/profile_images/529422070451433472/rBnqIhD9_normal.png'
    test_arr = np.array([[test1_id,test1_img, 0], [test2_id, test2_img, 0],\
            [test3_id, test3_img, 0]])
    test_data = pd.DataFrame(test_arr,\
            columns = ['tweet_id', 'profile_image_url_https', 'default_profile_image'])
    start = time.time()
    gender, age = GAC.predict(test_data)
    end = time.time()
    print(gender)
    print(age)
    print(end - start)
