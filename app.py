import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            subprocess.run(['streamlit', 'run', 'streamlit_app.py', '--', filepath])
            return redirect(url_for('success'))
        except Exception as e:
            return str(e)
    return "File type not allowed, please upload a CSV file."
@app.route('/success')
def success():
    return "File uploaded and processed successfully!"
if __name__ == '__main__':
    app.run(debug=True)
