from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            confidence_percent = round(confidence * 100, 2)
            
            if confidence_percent < 60:
                prediction_text = f"⚠️ Low Confidence Prediction: {label} ({confidence_percent}%)<br><small style='color:red;'>This prediction is uncertain. Please consult a dermatologist.</small>"
            else:
                prediction_text = f"Prediction: {label} ({confidence_percent}%)"

            return render_template("index.html", prediction=prediction_text, image=filename)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
