from exercise_imgs import *
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
from prediction_pipeline import predict, calc_distance
import numpy as np

ALLOWED_EXTENSIONS = {'mov', 'mp4'}
UPLOAD_FOLDER = './uploads'

if not os.path.exists('./uploads'):
    os.mkdir('./uploads')
    
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    
    
    
@app.route('/predict', methods=['POST'])
def get_prediction():
    exercise_name = request.form.get('exercise_name')
    video = request.files['video']
    
    active_img = np.array(Image.open(pose_coords.ground_truth_dict[exercise_name]['active_img']).convert('RGB'))
    rest_img = np.array(Image.open(pose_coords.ground_truth_dict[exercise_name]['rest_img']).convert('RGB'))
    
    pred_active = predict(active_img)
    pred_rest = predict(rest_img)
    
    if video and allowed_file(video.filename):
        filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    cam = cv2.VideoCapture(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    
    m_q_scores = np.empty(shape=(num_frames, 2))
    
    while(True):
        # reading from frame
        ret, frame = cam.read()
    
        # if video is still left continue creating images
        if ret:
            pred = predict(frame)             
            m_q_scores[frame, 0] = calc_distance(pred_active, pred)
            m_q_scores[frame, 1] = calc_distance(pred_rest, pred)
            
        else:
            break
  
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    
    score = 1.0 - m_q_scores.min(axis=0).mean()
    
    return jsonify({
        'movement_quality_score': score
    })

if __name__ == '__main__':
    app.run(debug=True)