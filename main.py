import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = "uploads"

classes_gender = ["男性", "女性"]
classes_age = ["未成年", "成人"]
num_classes_gender = len(classes_gender)
num_classes_age = len(classes_age)
image_size = 150

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_gender = load_model('./gender_model0709.h5')
model_age = load_model('./age_model0709.h5')

graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                img = image.load_img(filepath, target_size=(image_size,image_size))
                img = image.img_to_array(img)
                data = np.array([img])
                result_gender = model_gender.predict(data)[0]
                result_age = model_age.predict(data)[0]
                predicted_gender = result_gender.argmax()
                predicted_age = result_age.argmax()
                pred_answer = "この人は " + classes_age[predicted_age] + " の " + classes_gender[predicted_gender] + " です。"

                return render_template("index.html",answer=pred_answer)

        return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)