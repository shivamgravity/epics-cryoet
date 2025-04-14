from flask import Flask, render_template, request
import os, pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Map keywords to Google Drive links
drive_links = {
    '1': 'static/Images/apo-ferritin/Input.png',
    '_1': 'static/Images/apo-ferritin/Output.png',
    '2': 'static/Images/beta-amylase/Input.png',
    '_2': 'static/Images/beta-amylase/Output.png',
    '3': 'static/Images/beta-galactosidase/Input.png',
    '_3': 'static/Images/beta-galactosidase/Output.png',
    '4': 'static/Images/ribosome/Input.png',
    '_4': 'static/Images/ribosome/Output.png',
    '5': 'static/Images/thyroglobulin/Input.png',
    '_5': 'static/Images/thyroglobulin/Output.png',
    '6': 'static/Images/virus-like-particle/Input.png',
    '_6': 'static/Images/virus-like-particle/Output.png',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('file')

    folder_name = None
    output_imgs = set()

    for file in uploaded_files:
        if file.filename == '':
            continue

        # Extract folder name from relative file path
        parts = file.filename.split('/')
        if len(parts) > 1:
            folder_name = parts[0]
            break

    for keyword, link in drive_links.items():
        if keyword in folder_name:
            output_imgs.add(link)

    return render_template('result.html', output_imgs=list(output_imgs))

if __name__ == '__main__':
    app.run(debug=True)
