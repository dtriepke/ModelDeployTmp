from flask import Flask
import flask

app = Flask(__name__)

@app.route("/predict", methods=["GET", "POST"])
def pups():
    import joblib
    import numpy as np
    data = {"success": False}
    # get the request parameters
    model = joblib.load("models/ma_model.pcl")

    params = flask.request.json
    if (params == None):
        params = flask.request.args
    # if parameters are found, echo the msg parameter
    if (params is not None):
        response = params.get("raw_input")
        response = np.array(eval(response))
        data["response"] = model.predict(response)


        data["success"] = True
    # return a response in json format
    return flask.jsonify(data)

if __name__ == "main":
    app.run()


