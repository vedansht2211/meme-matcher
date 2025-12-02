from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
from matcher_core import MemeMatcherCore

app = Flask(__name__)

# Ensure absolute paths for robustness on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEME_FOLDER = os.path.join(BASE_DIR, 'memes_folder')
LEARNED_DATA = os.path.join(BASE_DIR, 'learned_data.json')

# Initialize the matcher
# Note: In a production environment with multiple workers, this would be loaded per worker.
matcher = MemeMatcherCore(meme_folder=MEME_FOLDER, learned_data_file=LEARNED_DATA)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/memes/<path:filename>')
def serve_meme(filename):
    return send_from_directory(matcher.meme_folder, filename)

@app.route('/debug')
def debug_info():
    """Helper to diagnose deployment issues"""
    try:
        meme_files = os.listdir(matcher.meme_folder) if os.path.exists(matcher.meme_folder) else []
        return jsonify({
            'base_dir': BASE_DIR,
            'meme_folder_path': matcher.meme_folder,
            'meme_folder_exists': os.path.exists(matcher.meme_folder),
            'meme_files_count': len(meme_files),
            'meme_files_sample': meme_files[:5],
            'loaded_memes_count': len(matcher.meme_features),
            'learned_data_count': len(matcher.learned_data),
            'cwd': os.getcwd()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        # Format is usually "data:image/jpeg;base64,......"
        if "," in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
            
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        match_name, score = matcher.process_frame(rgb)
        
        # Log the result to help debugging
        print(f"[DEBUG] Match: {match_name}, Score: {score}")
        
        return jsonify({
            'match_name': match_name,
            'score': float(score) if score is not None else 0.0
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get the PORT from Render, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # debug=False is safer for production, though we might want True for debugging right now
    app.run(debug=False, host='0.0.0.0', port=port)


