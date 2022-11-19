from flask import Flask,render_template,request
from Project_data.utils import HeartAttackDetection

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods = ["POST"])
def test():
    data = request.form 

    age=eval(data["age"])
    sex=eval(data["sex"])
    cp=eval(data["cp"])
    trestbps=eval(data["trestbps"])
    chol=eval(data["chol"])
    fbs=eval(data["fbs"])
    restecg=eval(data["restecg"])
    thalach=eval(data["thalach"])
    exang=eval(data["exang"])
    oldpeak=eval(data["oldpeak"])
    slope=eval(data["slope"])
    ca=eval(data["ca"])
    thal=eval(data["thal"])

    obj = HeartAttackDetection(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    result = obj.heartattack_prediction()
    return render_template("after.html",data = result)    

if __name__== "__main__":
    app.run(host="0.0.0.0",port=8080,debug=False)
    