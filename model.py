# remove warning message
# from statistics import StatisticsError
# import statistics
# from google.colab.patches import cv2_imshow
from os.path import splitext, basename
import numpy as np
import cv2
# import glob
from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import model_from_json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




# required library
# from local_utils import detect_lp

# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long


class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
            self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)


def getWH(shape):
    return np.array(shape[1::-1]).astype(float)


def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1-tl1, br2-tl2
    assert ((wh1 >= 0).all() and (wh2 >= 0).all())

    intersection_wh = np.maximum(np.minimum(
        br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area/union_area


def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)

    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels


def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T

        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H


def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)


def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

# Reconstruction function from predict value into plate crpoped from image


def reconstruct(I, Iresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2
    net_stride = 2**4
    side = ((208 + 40)/2)/net_stride

    # one line and two lines license plate size
    one_line = (470, 110)
    two_lines = (280, 200)

    Probs = Yr[..., 0]
    Affines = Yr[..., 2:]

    xx, yy = np.where(Probs > lp_threshold)
    # CNN input image size
    WH = getWH(Iresized.shape)
    # output feature map size
    MN = WH/net_stride

    vxx = vyy = 0.5  # alpha
    def base(vx, vy): return np.matrix(
        [[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T
    labels = []
    labels_frontal = []

    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        pts_frontal = np.array(B*base(vxx, vyy))

        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))

    final_labels = nms(labels, 0.1)
    final_labels_frontal = nms(labels_frontal, 0.1)

    # print(final_labels_frontal)
    assert final_labels_frontal, "No License plate is founded!"

    # LP size and type
    out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[
        0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

    TLp = []
    Cor = []
    if len(final_labels):
        final_labels.sort(key=lambda x: x.prob(), reverse=True)
        for _, label in enumerate(final_labels):
            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            ptsh = np.concatenate(
                (label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
            TLp.append(Ilp)
            Cor.append(ptsh)
    return final_labels, TLp, lp_type, Cor


def detect_lp(model, I, max_dim, lp_threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img
    w, h = (np.array(I.shape[1::-1], dtype=float)
            * factor).astype(int).tolist()
    Iresized = cv2.resize(I, (w, h))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)
    # print(Yr.shape)
    L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    return L, TLp, lp_type, Cor


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)





def preprocess_image(img, resize=False):
    # img = cv2.imread(image_path)
    # image_path = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(image_path, Dmax=608, Dmin=88):
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(
        wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    # fig = plt.figure(figsize=(12, 6))
    # grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    # fig.add_subplot(grid[0])
    # plt.axis(False)
    # plt.imshow(vehicle)
    # grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    # fig.add_subplot(grid[1])
    # plt.axis(False)
    # plt.imshow(LpImg[0])
    return vehicle, LpImg, cor


def Proccess_after_getplate(LpImg):
    if (len(LpImg)):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Applied inversed thresh_binary
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    # # visualize results
    # fig = plt.figure(figsize=(12, 7))
    # plt.rcParams.update({"font.size": 18})
    # grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    # plot_image = [plate_image, gray, blur, binary, thre_mor]
    # plot_name = ["plate_image", "gray", "blur", "binary", "dilation"]

    # for i in range(len(plot_image)):
    #     fig.add_subplot(grid[i])
    #     plt.axis(False)
    #     plt.title(plot_name[i])
    #     if i == 0:
    #         plt.imshow(plot_image[i])
    #     else:
    #         plt.imshow(plot_image[i], cmap="gray")

    # plt.savefig("threshding.png", dpi=300)
    return plate_image, binary, thre_mor


# Create sort_contours() function to grab the contour of each digit from left to right

def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# creat a copy version "test_roi" of plat_image to draw bounding box


# # Initialize a list which will be used to append charater image
# crop_characters = []

# # define standard width and height of character
# digit_w, digit_h = 30, 60
# data = []
# plate_cor = []
# heightRatio_arr = []
def Ocr_segment(cont, plate_image, binary, thre_mor):
    test_roi = plate_image.copy()
    crop_characters = []
# define standard width and height of character
    digit_w, digit_h = 30, 60
    data = []
    plate_cor = []
    heightRatio_arr = []
    for c in sort_contours(cont):

        (x, y, w, h) = cv2.boundingRect(c)
        aspectRatio = w / float(h)  # tỷ lệ rộng dài
        # tỷ lệ diện tích bao quanh
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(plate_image.shape[0])  # tỷ lệ chiều dài ký tự
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:

            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Sperate number and gibe prediction
            curr = thre_mor[y:y+h, x:x+w]
            curr = cv2.resize(curr, (30, 90), cv2.INTER_AREA)
            _, curr_num = cv2.threshold(
                curr, 99, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)
            curr_num = curr_num.reshape(90, 30, 1)
            heightRatio_arr.append(heightRatio)
            data.append((curr, (y, x)))
            plate_cor.append((int((x+w)/2), int((y+h)/2)))
            # print(x, y)
            # print((int((x+w)/2), int((y+h)/2)))
            # cv2_imshow(curr)
    return data, crop_characters, heightRatio

# print(plate_cor)
# new_sor = sorted(plate_cor , key=lambda k: [k[1], k[0]])
# print(new_sor)


def sortCharacters(heightRatio, crop_characters):
    crop = []
    if heightRatio < 0.6:
        for i in range(1, len(crop_characters), 2):
            crop.append(crop_characters[i])
        for i in range(0, len(crop_characters), 2):
            crop.append((crop_characters[i]))
        return crop
    else:
        return crop_characters
# try:
#   if statistics.mean(heightRatio_arr) < 0.5:
#     crop_characters = sortCharacters(crop_characters)
# except StatisticsError:
#   print("No number detected")


# print("Detect {} letters...".format(len(crop_characters)))
# fig = plt.figure(figsize=(10,6))
# plt.axis(False)
# plt.imshow(test_roi)
# plt.savefig('grab_digit_contour.png',dpi=300)
# Load model architecture, weight and labels



json_file = open(
    'MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(
    "License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load(
    'license_character_classes.npy')
print("[INFO] Labels loaded successfully...")




# pre-processing input images and pedict with model


def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform(
        [np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction


def out_put(crop_characters):
    # fig = plt.figure(figsize=(15, 3))
    # cols = len(crop_characters)
    # grid = gridspec.GridSpec(ncols=cols, nrows=1, figure=fig)

    final_string = ''
    for i, character in enumerate(crop_characters):
        # fig.add_subplot(grid[i])

        title = np.array2string(predict_from_model(character, model, labels))
        # plt.title('{}'.format(title.strip("'[]"), fontsize=20))
        final_string += title.strip("'[]")
        # plt.axis(False)
        # plt.imshow(character, cmap='gray')
    print("Hien thi bien so xe thu duoc la: ", final_string)
    return final_string
    #plt.savefig('final_result.png', dpi=300)


def predict(img):
    # test_image_path = "image_recognize.jpg"
    vehicle, LpImg, cor = get_plate(img)
    plate_image, binary, thre_mor = Proccess_after_getplate(LpImg)
    cont, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(plate_image, cont, -1, (0, 255, 0), 3)
    # cv2_imshow(plate_image)
    # cv2_imshow(binary)
    test_roi = plate_image.copy()

    data, crop_characters, heightRatio = Ocr_segment(
        cont, plate_image, binary, thre_mor)
    print("Detect {} letters...".format(len(crop_characters)))

    # if len(crop_characters) <= 7:
    #   mask_ero = np.ones((3, 3), np.uint8)
    #   mask_closing = np.ones((3, 3), np.uint8)
    #   img_erosion = cv2.erode(binary, mask_ero, iterations=1)
    #   img_dilation = cv2.dilate(img_erosion, mask_closing, iterations=1)
    #   cv2_imshow(img_dilation)
    #   cont, _  = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #   data, crop_characters, heightRatio =Ocr_segment(cont,plate_image,binary, thre_mor)
    #   print("Detect {} letters...".format(len(crop_characters)))
    #   cv2.drawContours(plate_image, cont, -1, (0, 255, 0), 3)
    #   cv2_imshow(plate_image)
    #   cv2_imshow(binary)
    
    
    # crop_characters = sortCharacters(heightRatio, crop_characters)
    
    
    # print(len(crop_characters))
    
    # fig = plt.figure(figsize=(10,6))
    # plt.axis(False)
    # plt.imshow(test_roi)
    # plt.savefig('grab_digit_contour.png',dpi=300)
    return out_put(crop_characters)
    

# sum_predict()
