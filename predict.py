# Prediction on external image...
from keras.models import load_model
 
# load model
import cv2 
import numpy as np

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9',
            36:'ka',37:'kha',38:'ga',39:'gha',40:'ṄA',41:'cha',42:'chha',43:'ja',44:'jha',45:'ÑA',46:'ta',47:'tha',48:'da',49:'dha',50:'Na',51:'ta',52:'tha',53:'da',54:'dha',55:'na',56:'pa',57:'pha',58:'ba',59:'bha',60:'ma',61:'ya',62:'ra',63:'la',64:'va',65:'sha',66:'sha',67:'sa',68:'ha',69:'ksh',70:'tra',71:'gya'}

model = load_model('HCR.h5')
img = cv2.imread(r'DA.jpg')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,440))

img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))


img_pred = word_dict[np.argmax(model.predict(img_final))]

cv2.putText(img, "Letter ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
cv2.imshow('Characters', img)


while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()