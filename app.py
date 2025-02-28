import cv2
import base64
import numpy as np 
from PIL import Image
from io import BytesIO
from pyngrok import ngrok
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from segment_anything import sam_model_registry, SamPredictor

ngrok.set_auth_token("")
public_url = ngrok.connect()
print(f"Public URL: {public_url}")

app = Flask(__name__)
CORS(app)

Sam = None

@app.route('/', methods=['GET'])
def check ():
    return jsonify({
        'message': 'Connection Secured'
    }), 200

@app.route('/sam/', methods=['POST'])
def segment ():
    try:
        data = request.get_json()

        if not data :
            return jsonify({
                'message': 'No data provided'
            }), 400
        else:
            print("Recieved data")
            
        global Sam
        if Sam == None:
            print("Initializing Predictor, ", end=" ")
            Sam = Init_predictor()
            Sam.model.to('cuda')
        try:  
            image_data = base64.b64decode(data['image'])
            image = Image.open(BytesIO(image_data))
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
            points = [np.array([point]) for point in data['points']]
        
        
            print("Masking Images, ", end=" ")
            masked_images = [Masked(Sam, image, point, True) for point in points]
            results = [image_array_to_base64(img) for img in masked_images]
        except Exception as e:
            print(e)

        print("Done")
        return jsonify({
            'images': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'message': f"Internal server error: {e}"
        }), 500

app.run(host='0.0.0.0', port=80, debug=True, use_reloader=False)
