import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in predictions.db
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model

with open(os.path.join('tmp', 'columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('tmp', 'pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('tmp', 'dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json

    obs_dict = request.get_json()
        
    #data validation
    error_validation = 0

    #check if the resquest has id and data
    if "observation_id" not in obs_dict:
        error_validation = 1
       
        return {'observation_id': None,
                "error": "observation_id"
               }
    else:
        _id = obs_dict['observation_id']
        

    if "data" not in obs_dict:
        error_validation = 1
        
        return {'observation_id': _id,
                "error": "data"
               }
    else:
        
        #columns's check
        valid_columns = {'age', 'sex', 'cp', 'trestbps', 'fbs', 'restecg', 'oldpeak', 'ca', 'thal'}

        keys = set(obs_dict['data'].keys())

        if len(valid_columns - keys) > 0: 
            missing = valid_columns - keys
            error_validation = 1
            
            return {'observation_id': _id,
                    "error": missing
                   }

        if len(keys - valid_columns) > 0: 
            extra = keys - valid_columns
            error_validation = 1
            
            return {'observation_id': _id,
                    "error": extra
                   }

    #check correct values 
    observation = obs_dict['data']

    valid_category_map = {
                            "sex": [0,1],
                            "cp": [0,1,2,3],
                            "fbs": [0,1],
                            "restecg": [0,1,2],
                            "ca": [0,1,2,3,4],
                            "thal": [0,1,2,3]
                         }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = value
                return {'observation_id': _id,
                        "error": key + ' ' + str(value)
                       }
     
    if observation.get("age") < 0 or observation.get("age") > 100:
        return {'observation_id': _id,
                "error": 'age' + ' ' + str(observation.get("age"))
                }
    
    if observation.get("trestbps") <= 10 or observation.get("trestbps") >= 500:
        return {'observation_id': _id,
                "error": 'trestbps' + ' ' + str(observation.get("trestbps"))
                }
    
    if observation.get("oldpeak") >= 12:
        return {'observation_id': _id,
                "error": 'oldpeak' + ' ' + str(observation.get("oldpeak"))
                }
                   
    if error_validation == 0:
        
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
        proba = pipeline.predict_proba(obs)[0, 1]
        prediction = pipeline.predict(obs)[0]
        response = {
                    'observation_id': _id,
                    "prediction": bool(prediction),
                    "probability": proba
                    }

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )        

    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
