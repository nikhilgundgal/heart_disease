import numpy as np
import pickle
import os
import json

model_file = os.path.join("Project_data","dt_heart_model.pkl")
json_file = os.path.join("Project_data","column_data.json")


class HeartAttackDetection():
    def __init__(self,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal


    def load_data(self):
        with open(model_file,"rb") as f:
           self.model = pickle.load(f)

        with open(json_file,"r") as j:
            self.json = json.load(j)  


    def heartattack_prediction(self):
        self.load_data()

        test_array = np.zeros(len(self.json["column"]))

        test_array[0]= self.age
        test_array[1]= self.sex
        test_array[2]= self.cp
        test_array[3]= self.trestbps
        test_array[4]= self.chol
        test_array[5]= self.fbs
        test_array[6]= self.restecg
        test_array[7]= self.thalach
        test_array[8]= self.exang
        test_array[9]= self.oldpeak
        test_array[10]= self.slope
        test_array[11]= self.ca
        test_array[12]= self.thal

        prediction = self.model.predict([test_array])[0]
        return prediction  
    


# age=63.0
# sex=1.0
# cp=3.0
# trestbps=145.0
# chol=233.0
# fbs=1.0
# restecg=0.0
# thalach=150.0
# exang=0.0
# oldpeak=2.3
# slope=0.0
# ca=0.0
# thal=1.0


# obj = HeartAttackDetection(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
# x = obj.heartattack_prediction()
# print(x)