#Genel
import sys, os, math, csv, random, time, datetime

#PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui
from tasarim import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox


#TensorFlow
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import get_file

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard

#OpenCv
import cv2

#Yardımcı
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import threading, webbrowser, signal


class App(QtWidgets.QMainWindow):
    def __init__(self): #Nesne oluşturuluyor
        super(App, self).__init__()
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QtGui.QIcon("assets\logo.png"))

        
        self.CATEGORIES = ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"]
        self.subdirs=[]
        self.dircounts=[]
        self.piccount=0
        self.test_o=0
        self.egitim_o=0
        self.validasyon_o=0
        self.dataset=""
        for path, subdirs, files in os.walk("csv_dataset"):
            for name in files:
                if(len(files)>0):
                    ds_adi=str(name)
                    o_t=str(time.ctime(os.path.getctime(os.path.join(path, name))))
                    g_t=str(time.ctime(os.path.getmtime(os.path.join(path, name))))
                    
                    self.ui.label_28.setText("Adı: "+ds_adi)
                    self.ui.label_29.setText("Oluşturulma tarihi: "+o_t)
                    self.ui.label_30.setText("Son güncelleme tarihi: "+g_t)
                    self.ui.label_39.setText("Adı: "+ds_adi)
                    self.ui.label_40.setText("Oluşturulma tarihi: "+o_t)
                    self.ui.label_41.setText("Son güncelleme tarihi: "+g_t)
                    self.ui.label_33.setText("Adı: "+ds_adi)
                    self.ui.label_34.setText("Oluşturulma tarihi: "+o_t)
                    self.ui.label_35.setText("Son güncelleme tarihi: "+g_t)
                    self.ui.label_43.setText("Adı: "+ds_adi)
                    self.ui.label_44.setText("Oluşturulma tarihi: "+o_t)
                    self.ui.label_45.setText("Son güncelleme tarihi: "+g_t)
                    
            
        self.yontem="Özel CNN ağı"
        self.epochs=10
        self.batch_size=64
        self.trasfer_model_path="MobileNetV2"
        self.Model_adi=""


        self.t1 = threading.Thread(target=self.tensorboard_log)
        self.test_model_path=""
        check=False
        for path, subdirs, files in os.walk("Models"):
            for name in files:
                if(not check):
                    self.ui.pushButton_6.setEnabled(True)
                    self.test_model_path="Models/"+name
                    check=True
                self.ui.comboBox_3.addItem(name) 
        

        if(self.test_model_path!=""):
            self.ui.label_18.setText("Şuanki model: "+self.test_model_path.replace("Models/",""))
            self.ui.pushButton_8.setEnabled(True)

        
        self.ui.pushButton.clicked.connect(self.dataset_folder)
        self.ui.pushButton_2.clicked.connect(self.show_folder_info)
        self.ui.pushButton_3.clicked.connect(self.split_dataset)
        self.ui.pushButton_4.clicked.connect(self.csv_veriseti)


        self.ui.comboBox.currentTextChanged.connect(self.yontem_sec)
        self.ui.comboBox_2.currentTextChanged.connect(self.tl_yontem)
        self.ui.radioButton.clicked.connect(self.radiobtn_durum)
        self.ui.radioButton_2.clicked.connect(self.radiobtn_durum)
        self.ui.radioButton_3.clicked.connect(self.radiobtn_durum)
        self.ui.radioButton_4.clicked.connect(self.radiobtn_durum)
        self.ui.pushButton_5.clicked.connect(self.egit)
        self.ui.pushButton_9.clicked.connect(self.model_ad)


        self.ui.comboBox_3.currentTextChanged.connect(self.test_model)
        self.ui.pushButton_6.clicked.connect(self.test)
        self.ui.pushButton_7.clicked.connect(self.tensorboard_log_thread)


        self.ui.pushButton_8.clicked.connect(self.model_donustur)


#Veriseti       
    def dataset_folder(self):
        dialog = QFileDialog()
        self.dataset = dialog.getExistingDirectory(self, 'Veri Seti Klasörü')
        
        self.piccount=0
        
        ctr=False
        ct=0
        for path, subdirs, files in os.walk(self.dataset):
            if(not(ctr)):
               self.subdirs=subdirs
               ctr=True
            #print(subdirs)
            for name in files:
                ct+=1
                self.piccount+=1
                #print(os.path.join(path, name))
            self.dircounts.append(ct)
            ct=0
        
        self.dircounts.pop(0)
        print("Dataset yol:", self.dataset)
        print("Resim sayisi: ", self.piccount)
        p_c=str(self.piccount)
        c_c=str(len(self.subdirs))
        self.ui.label_26.setText("Resim sayısı: "+p_c)
        self.ui.label_27.setText("Class sayısı: "+c_c)
        self.ui.label_37.setText("Resim sayısı: "+p_c)
        self.ui.label_38.setText("Class sayısı: "+c_c)
        self.ui.label_31.setText("Resim sayısı: "+p_c)
        self.ui.label_32.setText("Class sayısı: "+c_c)
        self.ui.label_36.setText("Resim sayısı: "+p_c)
        self.ui.label_42.setText("Class sayısı: "+c_c)

        if(self.piccount> 0):
            self.ui.pushButton_2.setEnabled(True)
            self.ui.pushButton_3.setEnabled(True)
      
    def show_folder_info(self):
        plt.bar(self.subdirs, self.dircounts, color ='blue', width = 0.4) 
        
        plt.title("Veri Seti İçeriği") 
        plt.xlabel("Sınıflar") 
        plt.ylabel("Örnek sayısı")
        
        plt.show()
        
    def split_dataset(self):
        self.test_o=int(self.ui.spinBox.text())
        
        self.egitim_o= 100 - int(self.ui.spinBox.text())
        
        # if(self.test_o+self.validasyon_o>100):
        #     print("Hata")
        #     msg = QMessageBox()
        #     msg.setWindowTitle("Tutorial on PyQt5")
        #     msg.setText("This is the main text!")
        #     msg.setIcon(QMessageBox.Critical)
        #     x = msg.exec_()
        #     self.ui.spinBox.setValue(1)
        #     self.split_dataset()
            
        
        self.ui.label_6.setText(str(math.floor(self.piccount*(int(self.ui.spinBox.text())/100))))
        self.ui.label_7.setText(str(self.piccount-(int(self.ui.label_6.text()))))

        self.ui.pushButton_4.setEnabled(True)
   
    def csv_veriseti(self):
        self.ui.label_24.setText("CSV Veri seti oluşturuluyor...")
        
        print("----- Resimler düzenlenmeye başlanıyor -----")
        
        with open('csv_dataset\csv_dataset.csv', 'w', newline='') as f:
            sutunadlari=['letter', 'pixels', 'status']
            ekleme=csv.DictWriter(f, fieldnames=sutunadlari)
            ekleme.writeheader()
            
            self.convert(0, ekleme)
    
    def convert(self, value, ekleme):
        
        
        print("----- İşlenen sınıf: "+ self.CATEGORIES[value]+" -----")
        
        for filename in os.listdir(self.dataset+"\\" + self.CATEGORIES[value]):
            full_path=self.dataset+"\\" + self.CATEGORIES[value] + "\\" + filename
                        
            print("Resim: ",full_path)
            
                    
            image = Image.open(full_path).convert('L')
            new_image = image.resize((48, 48))
            #new_image.save(filename+'.jpg') düzenlenmiş resmi kayıt etmek için
                        
            data = list(new_image.getdata())
                    
            durumlar=["Training", "Test"]
            durum= random.choices(durumlar, weights = [self.egitim_o, self.test_o])
            
            ekleme.writerow({'letter': value, 'pixels': str(data).strip('[]').replace(',', ''), 'status': str(durum).strip('[]').replace("'", "")})
         
        if(value!=len(self.CATEGORIES)-1):
            value+=1
            self.convert(value, ekleme)
        else:
            self.ui.label_24.setText("İşlem tamam")

    #Eğitim
    def radiobtn_durum(self):
        if self.ui.radioButton.isChecked():
            self.ui.groupBox_6.setEnabled(True)
        else:
            self.ui.groupBox_6.setEnabled(False)
        
        if self.ui.radioButton_3.isChecked():
            self.ui.groupBox_9.setEnabled(True)
        else:
            self.ui.groupBox_9.setEnabled(False)

    def yontem_sec(self):
        text=self.ui.comboBox.currentText()

        if(text=="Transfer öğrenme"):
            self.yontem="Transfer öğrenme"
            
            self.ui.groupBox_5.setEnabled(False)
            self.ui.groupBox_8.setEnabled(True)
        else:
            self.yontem="Özel CNN ağı"
            
            self.ui.groupBox_8.setEnabled(False)
            self.ui.groupBox_5.setEnabled(True)
        
        self.ui.label_16.setText("Yöntem: "+ self.yontem)
    
    def tl_yontem(self):
        self.ui.label_16.setText("Yöntem: "+ self.yontem+"("+str(self.ui.comboBox_2.currentText())+")")

    def model_ad(self):
        if(self.ui.lineEdit.text()!=""):
            self.ui.label_17.setText("Tam ad: "+str(self.ui.lineEdit.text())+"-"+str(self.yontem.replace(" ","-"))+".h5")
            self.Model_adi=str(self.ui.lineEdit.text())+"-"+str(self.yontem.replace(" ","-"))
            self.ui.pushButton_5.setEnabled(True)
        else:
            self.ui.pushButton_5.setEnabled(False)

    
    
    def egit(self):
        num_classes = 29
        print("=====EĞİTİM BAŞLIYOR======")
        if(self.yontem=="Özel CNN ağı"):
            if self.ui.radioButton.isChecked():
                self.epochs=int(self.ui.spinBox_2.text())
                self.batch_size=int(self.ui.spinBox_3.text())

            else:
                self.epochs=10
                self.batch_size=64

            print("Yöntem:", self.yontem)
            print("epoch:", self.epochs)
            print("Batchsize:", self.batch_size)
            print("----------------------------------------------------------------")
            
            self.ui.label_19.setText("Epochs: "+ str(self.epochs))
            self.ui.label_21.setText("Batch size: "+ str(self.batch_size))
            
            #Ön işlemler
            with open("csv_dataset\csv_dataset.csv") as f:
                content = f.readlines()

            lines = np.array(content)

            num_of_instances = lines.size
            print("-------------------------------------------")
            print("Resim sayısı: ",num_of_instances)
            print("Resimlerin boyutları: ",len(lines[1].split(",")[1].split(" ")))
            print("-------------------------------------------")

            x_train, y_train, x_test, y_test = [], [], [], []


            # Test ve eğitim verisinin transfer edilmesi
            for i in range(1,num_of_instances):
                
                letter, img, status = lines[i].split(",")
                
                val = img.split(" ")
                    
                pixels = np.array(val, 'float32')
                    
                letter = tensorflow.keras.utils.to_categorical(letter, num_classes)
                
                if 'Training' in status:
                    y_train.append(letter)
                    x_train.append(pixels)
                elif 'Test' in status:
                    y_test.append(letter)
                    x_test.append(pixels)

            # Eğtitim ve test kümelerinin diziye aktarılıyor
            x_train = np.array(x_train, 'float32')
            y_train = np.array(y_train, 'float32')
            x_test = np.array(x_test, 'float32')
            y_test = np.array(y_test, 'float32')

            x_train /= 255 # [0, 1] aralığına normalize etme işlemi
            x_test /= 255

            x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
            x_train = x_train.astype('float32')
            x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
            x_test = x_test.astype('float32')
            print("-------------------------------------------")
            print(x_train.shape[0], 'Eğitim Sayısı')
            print(x_test.shape[0], 'Test Sayısı')
            print("-------------------------------------------")
            
            model = Sequential()

            #1. Katamn
            model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
            model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

            #2. Katamn
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

            #3. Katamn
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

            model.add(Flatten())

            # Tam bağlantı katmanı
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(num_classes, activation='softmax'))
            
            #TesnorBoad
            log_dir = "TensorBoard_logs/" +self.Model_adi
            
            #board = TensorBoard(log_dir=log_dir, batch_size=64, histogram_freq=1, write_grads=False)
            board = TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True,write_images=True,update_freq='epoch',profile_batch=2,embeddings_freq=1)
            #tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            

            model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(), metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, y_test), callbacks=[board])


            model.save("Models/"+self.Model_adi+".h5")
            self.ui.label_47.setText("Model eğitildi ve kayıt edildi")
        
        
        
        if(self.yontem=="Transfer öğrenme"):
            self.trasfer_model=self.ui.comboBox_2.currentText()
            
            if self.ui.radioButton_3.isChecked():
                self.epochs=int(self.ui.spinBox_4.text())
                self.batch_size=int(self.ui.spinBox_5.text())
                self.trasfer_model=self.ui.comboBox_2.currentText()
            
            else:
                self.epochs=10
                self.batch_size=64
            
            print("Yöntem:", self.yontem)
            print("Model:", self.trasfer_model)
            print("epoch:", self.epochs)
            print("Batchsize:", self.batch_size)
            print("----------------------------------------------------------------")
            
            self.ui.label_19.setText("Epochs: "+ str(self.epochs))
            self.ui.label_21.setText("Batch size: "+ str(self.batch_size))


            #Ön işlemler
            
            if(self.trasfer_model!="InceptionV3"):
                IMAGE_SHAPE = (224, 224, 3)
            else:
                IMAGE_SHAPE = (299, 299, 3)
            
            CLASS_NAMES = np.array(self.CATEGORIES)
            # 20% validation set 80% training set
            image_generator = ImageDataGenerator(rescale=1/255, validation_split=self.test_o/100)
            
            train_data_gen = image_generator.flow_from_directory(directory=self.dataset, batch_size=self.batch_size, classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True, subset="training")
            test_data_gen = image_generator.flow_from_directory(directory=self.dataset, batch_size=self.batch_size, classes=list(CLASS_NAMES), target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]), shuffle=True, subset="validation")
            
            if(self.trasfer_model=="MobileNetV2"):
                model = MobileNetV2(input_shape=IMAGE_SHAPE)
            if(self.trasfer_model=="ResNet50"):
                model = ResNet50(input_shape=IMAGE_SHAPE)
            if(self.trasfer_model=="InceptionV3"):
                model = InceptionV3(input_shape=IMAGE_SHAPE)

            # remove the last fully connected layer
            model.layers.pop()
            # freeze all the weights of the model except the last 4 layers
            for layer in model.layers[:-4]:
                layer.trainable = False
            
            output = Dense(num_classes, activation="softmax")
            
            # connect that dense layer to the model
            output = output(model.layers[-1].output)
            model = Model(inputs=model.inputs, outputs=output)
            # print the summary of the model architecture
            #model.summary()
            # training the model using adam optimizer
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            
            training_steps_per_epoch = np.ceil(train_data_gen.samples / self.batch_size)
            validation_steps_per_epoch = np.ceil(test_data_gen.samples / self.batch_size)

            hist=model.fit_generator(train_data_gen, steps_per_epoch=training_steps_per_epoch, validation_data=test_data_gen, validation_steps=validation_steps_per_epoch, epochs=self.epochs)

            model.save("Models/"+self.Model_adi+".h5")
            self.ui.label_47.setText("Model eğitildi ve kayıt edildi")
            
        self.ui.comboBox_3.clear()
        cb_check=0
        for path, subdirs, files in os.walk("Models"):
            #print("Dosyalar: ", files)
            for name in files:
                #print(os.path.join(path, name))
                if(cb_check==0):
                    self.test_model_path="Models/"+name
                    self.ui.pushButton_6.setEnabled(True)
                    self.ui.label_18.setText("Şuanki model: "+ name)
                    self.ui.pushButton_8.setEnabled(True)
                    cb_check=cb_check+1
                self.ui.comboBox_3.addItem(name)
                
      
    #Test ve bilgi
            
    def test_model(self):
        self.test_model_path="Models/"+self.ui.comboBox_3.currentText()
        self.ui.label_18.setText("Şuanki model: "+ self.ui.comboBox_3.currentText())

    def tensorboard_log(self):
        webbrowser.open('http://localhost:6006/', new=1, autoraise=True)
            
        

        print("----Tensorboard değişiyor----")
        log_loc="C:\\Users\\DORUK57_2\\Desktop\\IsaretdiliAlgilama\\TensorBoard_logs\\"+self.ui.comboBox_3.currentText().replace(".h5","")
        os.system("tensorboard --logdir="+log_loc)

       
    def tensorboard_log_thread(self):
        self.t1.start()




    def model_bilgi(self, model, img):
        #Harlfer=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "Y", "Z"]
        
        img = image.load_img(img, grayscale=True, target_size=(48, 48))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x /= 255
        
        predictions = model.predict(x)
        harf=np.argmax(predictions[0])
        
        skorlar = predictions[0].argsort()[-len(predictions[0]):][::-1]

        max_score = 0.0
        
        for i in skorlar:
            score = predictions[0][i]
            if score > max_score:
                max_score = score
    
        return self.CATEGORIES[harf], round(max_score*100,2)
    
    def test(self):
        secilen_model=tf.keras.models.load_model(self.test_model_path)
        
        cap = cv2.VideoCapture(0)
        # cap.set(3, 1280)
        # cap.set(4, 720)
        while True:
            #kamera açlılıyor ve dondürülüyor
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            
            #çerçevenin bilgileri alınıyor
            height, width = img.shape[:2]
            
            #kırpılma alanı
            #print(height, width)
            x1, y1, x2, y2 = 313, 71, 600, 346 
            
            #kare çiziliyor
            cv2.rectangle(img , (313, 71), (600, 346), (0,255,0) , 2) #sol üst x-y sağ alt x-y
            
            #resim kırpılıyor ve gray oluor
            img_cropped = img[y1:y2,x1:x2]
            
            gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            
            #resim kaydediliyor
            cv2.imwrite("frame.png", gray)
            tahmin, orani=self.model_bilgi(secilen_model ,"frame.png")
            print("----------------------------")
            print("HARF:" + tahmin)
            #resime yazı yazılıyor
            cv2.putText(img, "Cerceve Bilgileri: "+str(width)+"x"+str(height), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(img, "Tahmin: "+str(tahmin)+" "+ str(orani)+"%", (315, 375), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            #Pencereler gösteriliyor
            cv2.imshow("Ana kamera", img)
            cv2.imshow("Islenen alan", gray)
            
            if cv2.waitKey(50) & 0xFF == ord('x'):
                break

        cap.release()
        cv2.destroyAllWindows()

    #Tesnorflow Lite Dönüşüm

    def model_donustur(self):
        tflite_model = tf.keras.models.load_model(self.test_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
        tflite_model = converter.convert()
        open("TFL_models/"+self.test_model_path.replace("Models/","")+".tflite", "wb").write(tflite_model)
        self.ui.label_48.setText("Model .tflite dosyasına cevirildi ve kayıtedildi")   
        
        
        
        
        
        
        
        
        
        
        
        






#Pencereyi göstermek için
def pencere():
    pencere=QtWidgets.QApplication(sys.argv)
    win=App()
    win.show()
    sys.exit(pencere.exec_())

pencere()
        