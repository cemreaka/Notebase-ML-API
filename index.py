from flask import Flask
from flask import request
from image_processing import predict_pdf

app = Flask(__name__)

@app.route('/', methods=['POST'])
def prediction():
    error = None
    if request.method == 'POST':
        uploaded_pdf = request.files['file'].read()
        result = predict_pdf(uploaded_pdf)
        return {"result":result}
    else:
        return {error:"Only POST is allowed."}