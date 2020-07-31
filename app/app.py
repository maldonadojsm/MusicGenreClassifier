# !/usr/bin/env python
# title           :app.py
# description     :Flask App
# author          :Sebastian Maldonado
# date            :7/22/2020
# version         :0.0
# usage           :SEE README.md
# notes           :Enter Notes Here
# python_version  :3.7.7
# conda_version   :4.8.2
# tf_version      :1.14
# =================================================================================================================

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tempfile
from utils import extract_audio, prepare_file, perform_inference

temp_directory = tempfile.gettempdir()

app = Flask(__name__)

app.config["AUDIO_UPLOADS"] = "../MusicGenreClassifier/app/static"
@app.route('/')
@app.route('/home')  # Root page of a website
def home():
    """
    :return: Render Home Page
    """
    return render_template('home.html')


@app.route('/classify_audio', methods=['GET', 'POST'])
def submit_music():
    """
    Processes inference for user's news article
    :return: Inference result
    """
    if request.method == "POST":

        if request.form['action'] == "Upload":
            f = request.files['audio_file']
            filename = secure_filename(f.filename)
            f.save(os.path.join(temp_directory, filename))
            test = "upload complete"
            return render_template("results.html", audio_path=test)
        # Capture Article URL

        if request.form['action'] == "Submit":
            url = request.form.get('youtube_url')
            extract_audio(url)
            data = prepare_file("audio_files/tf_audio.wav")
            prediction = perform_inference(data)
            return render_template('results.html', prediction=prediction)


    return render_template('classify_audio.html', title='Upload')


if __name__ == '__main__':
    app.run(debug=1)  # Debug mode activated
