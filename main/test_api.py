import flask
import pickle
import jsonschema
import pandas as pd
from dateutil import parser
from jsonschema import validate
from flask import request, jsonify, abort

app = flask.Flask(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "Age": {"type": "number"},
        "Job": {"type": "string"},
        "MaritalStatus": {"type": "string"},
        "EducationLevel": {"type": "string"},
        "CurrentCustomer": {"type": "number"},
        "UsesRelevantService":  {"type": "number"},
        "ContactMeans":  {"type": "string"},
        "ContactDay": {"type": "number"},
        "ContactMonth": {"type": "string"},
        "ContactsTotal": {"type": "number"},
        "DaysFromPrevAttempt": {"type": "number"},
        "PrevAttempts": {"type": "number"},
        "PrevOutcome": {"type": "string"},
        "CallStartTime": {"type": "string"},
        "CallEndTime": {"type": "string"},
    },
    "required": ["Age", "Job", "MaritalStatus", "EducationLevel", "CurrentCustomer", "UsesRelevantService",
                 "ContactMeans", "ContactDay", "ContactMonth", "ContactsTotal", "DaysFromPrevAttempt", "PrevAttempts",
                 "PrevOutcome", "CallStartTime", "CallEndTime"]
}


def validate_json(jsonData):
    try:
        validate(instance=jsonData, schema=SCHEMA)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True


def pre_process(data):

    # Replace nans with not known
    if data["Job"] == "null":
        data["Job"] = "not_known"
    if data["EducationLevel"] == "null":
        data["EducationLevel"] = "not_known"
    if data["ContactMeans"] == "null":
        data["ContactMeans"] = "not_known"
    if data["PrevOutcome"] == "null":
        data["PrevOutcome"] = "not_known"

    if data["DaysFromPrevAttempt"] == -1:
        data["DaysFromPrevAttempt"] = 0

    # Calculate call duration
    dt = parser.parse(data["CallEndTime"]) - parser.parse(data["CallStartTime"])
    data["CallDuration"] = dt.seconds

    del data["CallEndTime"], data["CallStartTime"]

    return data


def estimator(data):
    # Load model with all pipelines and return prediction
    filename = "../models/best_estimator.sav"
    model = pickle.load(open(filename, "rb"))
    return model.predict(data)[0]


@app.route('/predict', methods=['POST'])
def create_task():
    # Check if data follow schema
    if not validate_json(request.json):
        abort(400)

    # Initial pre processing
    data = pre_process(request.json)

    return jsonify({'Convert': int(estimator(pd.DataFrame.from_dict([data])))})


if __name__ == '__main__':
    app.run(debug=True)

