from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory
import os
import uuid
import base64
import cv2
import numpy as np
from app.utils import allowed_file, process_and_predict
from src.predict import load_models

main_bp = Blueprint('main', __name__)

@main_bp.record
def on_load(state):
    """Proactive cache initialization of both Neural networks preventing lag."""
    load_models()

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    # Detects toggle context (Digit / Character logic gate)
    mode = request.form.get('mode', 'digit')
    
    # Mode 1: Vector Canvas Processing
    if 'image_base64' in request.form:
        img_data = request.form['image_base64'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        result, error = process_and_predict(img, mode)
        if error: return jsonify({'error': error}), 500
            
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': f"{result['confidence'] * 100:.2f}%",
            'top_3': result['top_3'],
            'low_confidence': result['confidence'] < 0.80
        })
        
    # Mode 2: Physical File Upload Framework
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'Format Error: Nil stream.'}), 400
            
        if allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(filepath)
            result, error = process_and_predict(filepath, mode)
            
            if error:
                if os.path.exists(filepath): os.remove(filepath)
                return jsonify({'error': error}), 500
                
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'confidence': f"{result['confidence'] * 100:.2f}%",
                'top_3': result['top_3'],
                'low_confidence': result['confidence'] < 0.80,
                'image_url': f"/uploads/{unique_filename}"
            })
            
    return jsonify({'error': 'System Architecture declined request inputs.'}), 400

@main_bp.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], name)
