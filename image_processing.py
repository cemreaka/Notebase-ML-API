from pdf2image import convert_from_bytes
import numpy as np
import cv2
import keras
from imutils.perspective import four_point_transform
from keras.models import model_from_json


def crop_image(img):
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    image = cv2.copyMakeBorder(img_colored, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, value = 0)
    img_reshaped = cv2.resize(image, (900,900))
    dst = cv2.GaussianBlur(img_reshaped,(15,15),cv2.BORDER_DEFAULT)
    hh, ww = dst.shape[:2]
    lower = np.array([50, 50, 50])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(dst, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(img_reshaped, img_reshaped, mask=morph)
    edges= cv2.Canny(morph, 50,200)
    contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
    largest_item= sorted_contours[0]
    result_copy = result.copy()
    contour_mask = np.zeros_like(result_copy, dtype='uint8')
    contour_mask_img = cv2.fillPoly(contour_mask, pts = [largest_item], color=(255, 255, 255))
    final_image = cv2.bitwise_and(result_copy, result_copy, mask = cv2.cvtColor(contour_mask_img, cv2.COLOR_BGR2GRAY))
    for contour in sorted_contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        doc_cnts = approx
        if len(approx) == 4:
            break
    if(len(doc_cnts) != 4):
        warped = img_colored
    else:
        warped = four_point_transform(final_image, doc_cnts.reshape(4, 2))
    img_reshaped = cv2.resize(warped, (400,400))
    img_gray = cv2.cvtColor(img_reshaped, cv2.COLOR_BGR2GRAY)
    np_image = np.asarray(img_gray)
    np_final = np.expand_dims(np_image, axis = 2)   
    return np_final

# def pdf_to_image(pdf):
#     images = convert_from_path(pdf, fmt='jpg', poppler_path='C:\\Program Files\\poppler-0.68.0\\bin')
#     for page_count in range(len(images)):
#        images[page_count].save('temp_files\\page_'+str(page_count+1)+'.jpg', 'JPEG')

# def predict_pdf(pdf):
#     model = keras.models.load_model('model.h5')
#     pdf_to_image(pdf)
#     folder_name = "temp_files\\"
#     counter = 0
#     sum = 0
#     for filename in os.listdir(folder_name):
#         counter = counter+1
#         img = cv2.imread(os.path.join(folder_name,filename),0)
#         if img is not None:
#             Y = image.img_to_array(crop_image(img))
#             X = np.expand_dims(Y,axis=0)
#             val = model.predict(X)
#             sum = sum + val[0][0]
#             print(val[0][0])
#             os.remove(str(folder_name)+str(filename))
#     result = sum / counter
#     print(result)
#     if result >= 0.4:
#         return True
#     return False

def predict_pdf(pdf):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model_weights.h5")
    images = convert_from_bytes(pdf, fmt='jpg', poppler_path='poppler-0.68.0\\bin')
    counter = 0
    sum = 0
    for image in images:
        counter = counter+1
        img = np.array(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img_gray is not None:
            Y = crop_image(img_gray)
            X = np.expand_dims(Y,axis=0)
            val = model.predict(X)
            sum = sum + round(val[0][0])
    result = sum / counter
    if result >= 0.4:
        return True #valid
    return False #invalid


# img2 = cv2.imread('Dataset\\train\\valid\\eng_NA_066.jpg',0)
# img3 = cv2.imread('2-4.jpg',0)
# img4 = cv2.imread('Dataset\\test\\valid\\IMG_1293.JPG',0)
# final_image2 = crop_image(img2)
# final_image3 = crop_image(img3)
# final_image4 = crop_image(img4)
# cv2.imshow('Result1',final_image2)
# cv2.imshow('Result2',final_image3)
# cv2.imshow('Result3',final_image4)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# res1 = predict_pdf('Y18070006007_exp6.pdf')
# if res1:
#     print('Valid')
# else: 
#     print('Invalid')

# res2 = predict_pdf('EEE3132_Experiment_2.pdf')
# if res2:
#     print('Valid')
# else: 
#     print('Invalid')