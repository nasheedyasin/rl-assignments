from flask import Flask,render_template,request,jsonify
from flask_cors import CORS
# Use this if you want to check result on non trainer model
# from chatnotraining import get_response


#   use this for trained model
from chat import get_response

app = Flask(__name__)
CORS(app)
step =1

# Checking the turn first bot and persuader

# @app.post(/mymessage)
# def displayallhistory():
#   return history()




@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer":response}
    return jsonify(message)

if __name__== "__main__":
    app.run(debug=True)