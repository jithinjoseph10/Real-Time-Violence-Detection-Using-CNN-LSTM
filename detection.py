# detection.py
import cv2
import numpy as np
import keras
from keras.applications import VGG16
from keras.models import Model
from keras import backend as K
import final as f
from termcolor import colored

def real_time_detection():
    print("Real-Time Detection")
    f.start(0, 'nil')

def upload_video():
    filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if filepath:
        print("Uploaded Video:", filepath)
        filename = filepath.split('/')
        fname = filename[-1]
        text_label = tk.Label(window, text=fname + " Uploaded")
        text_label.pack()
        detect(1, filepath)

def detect(n, filepath):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('train.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    model = keras.models.load_model('model/vlstm_new.h5')
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input, outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_size = 224
    num_channels = 3
    _images_per_file = 20

    if n == 1:
        cap = cv2.VideoCapture(filepath)
    else:
        cap = cv2.VideoCapture(0)

    count = 0
    images = []
    shape = (_images_per_file, img_size, img_size, 3)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        count += 1
        if frame_count == ret:
            break
        else:
            if count <= _images_per_file:
                RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = cv2.resize(RGB_img, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
                images.append(res)
            else:
                resul = np.array(images)
                resul = (resul / 255.).astype(np.float16)
                transfer_values = image_model_transfer.predict(resul)
                inp = np.array(transfer_values)[np.newaxis, ...]
                pred = model.predict(inp)
                res = np.argmax(pred[0])
                count = 0
                images = []

                if res == 0:
                    print("\n\n" + colored('VIOLENT', 'red') + " Video with confidence: " + str(round(pred[0][res] * 100, 2)) + " %")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        if conf < 50:
                            if Id == 1:
                                Id = "Aleena"
                            elif Id == 2:
                                Id = "Deepthi"
                            elif Id == 3:
                                Id = "Chaitra"
                            elif Id == 4:
                                Id = "Abhi"
                        else:
                            Id = "unknown"
                        cv2.putText(frame, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)
                else:
                    print("\n\n" + colored('NON-VIOLENT', 'green') + " Video with confidence: " + str(round(pred[0][res] * 100, 2)) + " %")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
                        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        if conf < 50:
                            if Id == 1:
                                Id = "Aleena"
                            elif Id == 2:
                                Id = "Deepthi"
                            elif Id == 3:
                                Id = "Chaitra"
                            elif Id == 4:
                                Id = "Abhi"
                        else:
                            Id = "unknown"
                        cv2.putText(frame, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)

                if frame is not None:
                    cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord("q"):
                    break

    cv2.destroyAllWindows()
