from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
from matcher_core import MemeMatcherCore

app = Flask(__name__)

# Initialize the matcher
# Note: In a production environment with multiple workers, this would be loaded per worker.
matcher = MemeMatcherCore()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/memes/<path:filename>')
def serve_meme(filename):
    return send_from_directory(matcher.meme_folder, filename)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        # Format is usually "data:image/jpeg;base64,......"
        header, encoded = image_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process
        match_name, score = matcher.process_frame(rgb)
        
        return jsonify({
            'match_name': match_name,
            'score': float(score) if score is not None else 0.0
        })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    # Get the PORT from Render, default to 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    # debug=False is safer for production
    app.run(debug=False, host='0.0.0.0', port=port)
