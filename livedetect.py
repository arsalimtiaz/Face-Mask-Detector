from keras.models import load_model
import cv2
import numpy as np

print("Import Successful")

model = load_model('checkpoint/model-020.model')

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = {0: 'NO MASK', 1: 'MASK'}
color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}
gray = None

while True:

    ret, img = source.read()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        print(labels_dict[label])

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.putText(img, labels_dict[label], (x + 10, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Mask Detector', img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
source.release()
