import base64
import numpy as np
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from sam import Init_predictor, image_array_to_base64, Masked

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def check ():
    return jsonify({
        'message': 'Connection Secured'
    }), 200

@app.route('/sam/', methods=['POST'])
def segment ():
    try:
        data = request.json()

        if not data :
            return jsonify({
                'message': 'No data provided'
            }), 400

        image_data = base64.b64decode(data['image'])
        image = Image.open(BytesIO(image_data))
        points = [np.array(point) for point in data['points']]
        sam = Init_predictor()

        sam.to('cuda')
        masked_images = [Masked(sam, image, point, mask_it=True)]
        results = [image_array_to_base64(img) for img in masked_images]

        return jsonify({
            'images': results
        }), 200

    except Exception as e:
        return jsonify({
            'message': f"Internal server error: {error}"
        }), 500

app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
