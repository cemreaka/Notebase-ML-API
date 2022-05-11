from flask import Flask, json
from flask import request
from image_processing import predict_pdf
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['POST'])
@cross_origin()
def prediction():
    error = None
    if request.method == 'POST':
        uploaded_pdf = request.files['file'].read()
        result = predict_pdf(uploaded_pdf)
        print(result)
        return json.dumps(result)
    else:
        return {error:"Only POST is allowed."}

if __name__ == "__main__":
    app.run()