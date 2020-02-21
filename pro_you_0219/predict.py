import os

import cv2
import dlib
from PIL import Image
import numpy as np
import math

# 업로드 경로(폴더)
UPLOAD_PATH = "static/image/"

PREDICTOR_PATH = "static/landmarks/shape_predictor_68_face_landmarks.dat"

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class face_classification:


    # 초기화
    def __init__(self, model, sess, graph, filename):
    #def __init__(self, model, sess, graph):
        self.model = model
        self.graph = graph
        self.sess = sess
        self.filename = filename

    def annotate_landmarks(self, im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

    def get_landmarks(self, im):
        rects = detector(im, 1)

        return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    def read_im_and_landmarks(self, fname):
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                             im.shape[0] * SCALE_FACTOR))
        s = self.get_landmarks(im)

        return im, s

    def cut_eyes(self, im, landmarks):
        im = im[landmarks[37][0, 1] + 1:landmarks[40][0, 1] + 1, landmarks[37][0, 0] - 2:landmarks[38][0, 0] + 3, :]

        return im

    def select_index_to_use(self, index_dark, index_medium, index_light):
        max = 0
        name_of_index = ""
        index_to_use = []

        if len(index_dark) > len(index_medium):
            max = len(index_dark)
            index_to_use = index_dark
            name_of_index = "index_dark"
        else:
            max = len(index_medium)
            index_to_use = index_medium
            name_of_index = "index_medium"

        if max < len(index_light):
            print("index_light")
            index_to_use = index_light
        else:
            print(name_of_index)

        return index_to_use

    def season_matching(self, skin_tone, eye_brightness):
        if skin_tone == "warm":
            if eye_brightness == "dark":
                return "autumn"
            else:
                return "spring"
        else:
            if eye_brightness == "dark":
                return "winter"
            else:
                return "summer"

    def pccs_finder(self, season, s, v, spring_list, summer_list, autumn_list, winter_list):
        i = 0
        min = 2000
        skin_type = ""

        if season == "spring":
            while i < len(spring_list):
                distance = math.sqrt((s - spring_list[i][0]) ** 2 + (v - spring_list[i][1]) ** 2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + spring_list[i][2] + " : " + str(distance))
                    skin_type = spring_list[i][2]

                i += 1

        elif season == "summer":
            while i < len(summer_list):
                distance = math.sqrt((s - summer_list[i][0]) ** 2 + (v - summer_list[i][1]) ** 2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + summer_list[i][2] + " : " + str(distance))
                    skin_type = summer_list[i][2]

                i += 1

        elif season == "autumn":
            while i < len(autumn_list):
                distance = math.sqrt((s - autumn_list[i][0]) ** 2 + (v - autumn_list[i][1]) ** 2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + autumn_list[i][2] + " : " + str(distance))
                    skin_type = autumn_list[i][2]

                i += 1

        else:
            while i < len(winter_list):
                distance = math.sqrt((s - winter_list[i][0]) ** 2 + (v - winter_list[i][1]) ** 2)
                if min > distance:
                    min = distance
                    print("Calculating... " + winter_list[i][2] + " : " + str(distance))
                    skin_type = winter_list[i][2]

                i += 1

        return skin_type

    def before_predict(self):
        # 첫번째
        file_path = "static/image/"
        full_file_name = os.path.splitext(self.filename)
        file_name = full_file_name[0].lower()
        img_type = full_file_name[1].lower()  # 원본파일 확장자

        im, landmarks = self.read_im_and_landmarks(file_path + file_name + img_type)

        color_location1 = ((landmarks[54] + landmarks[11] + landmarks[45]) / 3).astype(int)  # left cheek
        color_location2 = ((landmarks[48] + landmarks[4] + landmarks[36]) / 3).astype(int)  # right cheek

        rgb1 = im[color_location1[0, 1], color_location1[0, 0]][2], im[color_location1[0, 1], color_location1[0, 0]][1], \
               im[color_location1[0, 1], color_location1[0, 0]][0]
        rgb2 = im[color_location2[0, 1], color_location2[0, 0]][2], im[color_location2[0, 1], color_location2[0, 0]][1], \
               im[color_location2[0, 1], color_location2[0, 0]][0]
        rgb = ((int(rgb1[0]) + int(rgb2[0])) / 2, (int(rgb1[1]) + int(rgb2[1])) / 2, (int(rgb1[2]) + int(rgb2[2])) / 2)

        cv2.putText(im, str("Target1"), (color_location1[0, 0] + 5, color_location1[0, 1]),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255))
        cv2.circle(im, (color_location1[0, 0], color_location1[0, 1]), 3, color=(0, 0, 255), thickness=-1)

        cv2.putText(im, str("Target2"), (color_location2[0, 0] + 5, color_location2[0, 1]),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255))
        cv2.circle(im, (color_location2[0, 0], color_location2[0, 1]), 3, color=(0, 0, 255), thickness=-1)

        im_annotated = self.annotate_landmarks(im, landmarks)

        lab_colors = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

        a_value1 = lab_colors[color_location1[0, 1], color_location1[0, 0]][1]  # a
        a_value2 = lab_colors[color_location2[0, 1], color_location2[0, 0]][1]

        a_value = (int(a_value1) + int(a_value2)) / 2

        b_value1 = lab_colors[color_location1[0, 1], color_location1[0, 0]][2]  # b
        b_value2 = lab_colors[color_location2[0, 1], color_location2[0, 0]][2]

        b_value = (int(b_value1) + int(b_value2)) / 2

        skin_ab_value = []

        info = []
        info.append(a_value)
        info.append(b_value)

        skin_ab_value.append(info)

        # 첫번째 결과 = 웜, 쿨
        result1 = self.predict_food(skin_ab_value)

        # 두번째
        cut_eyes = self.cut_eyes(im, landmarks)
        gray = cv2.cvtColor(cut_eyes, cv2.COLOR_BGR2GRAY)
        etval, thresholded = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        index_dark = []
        i = 0

        while i < cut_eyes.shape[0]:
            j = 0
            index_of_black = []
            while j < cut_eyes.shape[1]:
                if thresholded[i][j] == 0:
                    index_of_black.append(i)
                    index_of_black.append(j)
                    index_dark.append(index_of_black)
                    index_of_black = []
                j += 1
            i += 1

        etval, thresholded = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        i = 0

        while i < cut_eyes.shape[0]:
            j = 0
            index_deleted = []
            while j < cut_eyes.shape[1]:
                if thresholded[i][j] == 0:
                    index_deleted.append(i)
                    index_deleted.append(j)
                    if index_deleted in index_dark:
                        index_dark.remove(index_deleted)
                    index_deleted = []
                j += 1
            i += 1

        etval, thresholded = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        index_medium = []
        etval, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        i = 0

        while i < cut_eyes.shape[0]:
            j = 0
            index_deleted = []
            while j < cut_eyes.shape[1]:
                if thresholded[i][j] == 0:
                    index_deleted.append(i)
                    index_deleted.append(j)
                    if index_deleted in index_medium:
                        index_medium.remove(index_deleted)
                    index_deleted = []
                j += 1
            i += 1

        etval, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        index_light = []
        i = 0

        while i < cut_eyes.shape[0]:
            j = 0
            index_of_black = []
            while j < cut_eyes.shape[1]:
                if thresholded[i][j] == 0:
                    index_of_black.append(i)
                    index_of_black.append(j)
                    index_light.append(index_of_black)
                    index_of_black = []
                j += 1
            i += 1

        etval, thresholded = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
        i = 0

        while i < cut_eyes.shape[0]:
            j = 0
            index_deleted = []
            while j < cut_eyes.shape[1]:
                if thresholded[i][j] == 0:
                    index_deleted.append(i)
                    index_deleted.append(j)
                    if index_deleted in index_light:
                        index_light.remove(index_deleted)
                    index_deleted = []
                j += 1
            i += 1

        index_to_use = self.select_index_to_use(index_dark, index_medium, index_light)

        eye_lab = cv2.cvtColor(cut_eyes, cv2.COLOR_RGB2LAB)
        eye_l_value = []
        i = 0
        while i < len(index_to_use):
            eye_l_value.append(eye_lab[index_to_use[i][0], index_to_use[i][1], 0])
            i += 1

        i = 0
        while i < int(len(eye_l_value) * 15 / 85):
            eye_l_value.append(0)
            i += 1

        eye_l_value = np.array(eye_l_value)
        std = math.sqrt(np.sum((eye_l_value - np.mean(eye_l_value)) ** 2) / (eye_l_value.size))

        eye_brightness = ""

        if std < 38.28:
            eye_brightness = "dark"
        else:
            eye_brightness = "light"

        season = self.season_matching(result1, eye_brightness)

        color_info = [[0.87464539937997 * 255, 181, 'vivid'],
                      [0.6920954876849125 * 255, 207, 'bright'],
                      [0.8348038859274655 * 255, 170, 'strong'],
                      [0.8901501134915047 * 255, 137, 'deep'],
                      [0.3300064593086124 * 255, 221, 'light'],
                      [0.39800595815638706 * 255, 182, 'soft'],
                      [0.5135643825656158 * 255, 142, 'dull'],
                      [0.6815737217178439 * 255, 93, 'dark'],
                      [0.11073418459625402 * 255, 226, 'pale'],
                      [0.22483614835441365 * 255, 110, 'grayish'],
                      [0.26913149326755986 * 255, 58, 'dark_grayish'],
                      [0.20022609113608203 * 255, 186, 'light_grayish']]

        spring_list = []
        spring_list.append(color_info[0])
        spring_list.append(color_info[1])
        spring_list.append(color_info[4])
        spring_list.append(color_info[8])

        summer_list = []
        summer_list.append(color_info[4])
        summer_list.append(color_info[8])
        summer_list.append(color_info[5])
        summer_list.append(color_info[6])
        summer_list.append(color_info[7])
        summer_list.append(color_info[9])
        summer_list.append(color_info[10])
        summer_list.append(color_info[11])

        autumn_list = []
        autumn_list.append(color_info[3])
        autumn_list.append(color_info[5])
        autumn_list.append(color_info[6])
        autumn_list.append(color_info[9])
        autumn_list.append(color_info[3])
        autumn_list.append(color_info[7])

        winter_list = []
        winter_list.append(color_info[0])
        winter_list.append(color_info[2])
        winter_list.append(color_info[3])
        winter_list.append(color_info[7])
        winter_list.append(color_info[10])

        skin_v_value = max(rgb[0], rgb[1], rgb[2])
        skin_s_value = (1 - min(rgb[0], rgb[1], rgb[2]) / skin_v_value) * 255

        pccs = self.pccs_finder(color_info, skin_s_value, skin_v_value, spring_list, summer_list, autumn_list, winter_list)

        return result1, pccs, season

    # 모델을 통해 사진 속의 음식 예측 후 분류
    def predict_food(self, skin_ab_value):
        value = self.model.predict(skin_ab_value)

        if value == 0:
            return "warm"
        else:
            return "cool"

    # 예측된 음식명, 확률 가져오기
    def get_predicted(self):
        return self.before_predict()