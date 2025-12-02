import cv2
import mediapipe as mp
import numpy as np
import os
import json

# --- CONFIGURATION & CONSTANTS ---
MEME_FOLDER = "memes_folder"
LEARNED_DATA_FILE = "learned_data.json"

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

class MemeMatcherCore:
    def __init__(self, meme_folder=MEME_FOLDER, learned_data_file=LEARNED_DATA_FILE):
        self.meme_folder = meme_folder
        self.learned_data_file = learned_data_file
        self.meme_features = {}
        self.learned_data = []
        
        # Initialize MediaPipe instances for reuse
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Load data
        self.load_memes()
        self.load_learned_data()

    def get_face_features(self, face_landmarks):
        if not face_landmarks: return None
        landmarks = face_landmarks[0].landmark
        def dist(i1, i2):
            return np.linalg.norm(np.array([landmarks[i1].x, landmarks[i1].y]) - 
                                  np.array([landmarks[i2].x, landmarks[i2].y]))

        face_height = dist(10, 152)
        if face_height == 0: face_height = 1.0
        
        features = np.array([
            dist(78, 308) / face_height, # Mouth width
            dist(13, 14) / face_height,  # Mouth open
            (landmarks[78].y + landmarks[308].y) / 2 - landmarks[0].y, # Smile
            dist(105, 159) / face_height, # L Brow
            dist(334, 386) / face_height, # R Brow
            dist(159, 145) / face_height, # L Eye
            dist(386, 374) / face_height, # R Eye
            dist(13, 152) / face_height   # Jaw
        ])
        return features

    def get_hand_features(self, single_hand_landmarks):
        if not single_hand_landmarks: return None
        landmarks = single_hand_landmarks.landmark
        def dist(i1, i2):
            if isinstance(i1, int): p1 = landmarks[i1]
            else: p1 = i1
            if isinstance(i2, int): p2 = landmarks[i2]
            else: p2 = i2
            return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

        # Finger extensions
        thumb_ext = dist(4, 17) / dist(2, 17)
        index_ext = dist(8, 0) / dist(5, 0)
        middle_ext = dist(12, 0) / dist(9, 0)
        ring_ext = dist(16, 0) / dist(13, 0)
        pinky_ext = dist(20, 0) / dist(17, 0)
        
        # Orientation
        hand_dir_x = landmarks[9].x - landmarks[0].x
        hand_dir_y = landmarks[9].y - landmarks[0].y
        
        # Gestures
        is_pointing = (index_ext > 1.5) and (middle_ext < 1.2)
        is_fist = (index_ext < 1.2) and (middle_ext < 1.2)
        is_open = (index_ext > 1.5) and (middle_ext > 1.5)
        
        features = np.array([
            thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext,
            hand_dir_x, hand_dir_y,
            1.0 if is_pointing else 0.0,
            1.0 if is_fist else 0.0,
            1.0 if is_open else 0.0
        ])
        return features

    def load_memes(self):
        print("\n=== Loading Memes... ===")
        if not os.path.isdir(self.meme_folder):
            print(f"Error: Folder '{self.meme_folder}' not found.")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        meme_files = [f for f in os.listdir(self.meme_folder) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        for filename in meme_files:
            try:
                img_path = os.path.join(self.meme_folder, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                f_res = self.face_mesh.process(rgb)
                h_res = self.hands.process(rgb)
                
                self.meme_features[filename] = {
                    'face': self.get_face_features(f_res.multi_face_landmarks),
                    'hand': self.get_hand_features(h_res.multi_hand_landmarks[0]) if h_res.multi_hand_landmarks else None,
                    # We don't store the full image in memory for the web app to save RAM, 
                    # we'll serve it from disk
                }
                print(f"   Loaded: {filename}")
            except Exception as e: 
                print(f"Failed to load {filename}: {e}")

    def load_learned_data(self):
        if os.path.exists(self.learned_data_file):
            try:
                with open(self.learned_data_file, 'r') as f:
                    data = json.load(f)
                # Convert lists back to numpy arrays
                self.learned_data = []
                for item in data:
                    entry = item.copy()
                    if entry['face'] is not None: entry['face'] = np.array(entry['face'])
                    if entry['hand'] is not None: entry['hand'] = np.array(entry['hand'])
                    self.learned_data.append(entry)
                print(f"[INFO] Loaded {len(self.learned_data)} learned examples.")
            except Exception as e:
                print(f"[WARNING] Could not load learned data: {e}")
                self.learned_data = []
        else:
            self.learned_data = []

    def calculate_similarity(self, live_face, live_hand, target_face, target_hand):
        face_score = 0.0
        hand_score = 0.0
        
        if live_face is not None and target_face is not None:
            dot = np.dot(live_face, target_face)
            norm = np.linalg.norm(live_face) * np.linalg.norm(target_face)
            if norm > 0: face_score = (dot / norm + 1) / 2
            
        if live_hand is not None and target_hand is not None:
            dot = np.dot(live_hand, target_hand)
            norm = np.linalg.norm(live_hand) * np.linalg.norm(target_hand)
            if norm > 0: hand_score = (dot / norm + 1) / 2
        
        # Adaptive weighting
        if target_face is not None and target_hand is not None:
            return 0.5 * face_score + 0.5 * hand_score if live_hand is not None else 0.8 * face_score + 0.2 * hand_score
        elif target_face is not None:
            return face_score
        elif target_hand is not None:
            return hand_score if live_hand is not None else 0.0
        return 0.0

    def find_best_match(self, live_face, live_hand):
        best_match = None
        best_score = -1.0
        
        # 1. Check Original Memes
        for filename, features in self.meme_features.items():
            score = self.calculate_similarity(live_face, live_hand, features['face'], features['hand'])
            if score > best_score:
                best_score = score
                best_match = filename
                
        # 2. Check Learned Examples
        for item in self.learned_data:
            score = self.calculate_similarity(live_face, live_hand, item['face'], item['hand'])
            score *= 1.2  # Boost
            
            if score > best_score:
                best_score = score
                best_match = item['filename']
        
        return best_match, best_score

    def process_frame(self, frame_rgb):
        # Process a single frame and return features
        # Note: We create new instances here or use a lock if we were multithreaded, 
        # but for a simple flask app, we might just re-use if single worker.
        # For safety in Flask (threaded), it's better to process in the request context 
        # or use a pool. However, MediaPipe objects are not thread safe.
        # For this simple version, we will instantiate a fresh processor for the request 
        # OR use a global lock. Let's try instantiating fresh for robustness first, 
        # or just use the ones we have if we run with 1 worker.
        
        # Actually, let's just use the ones we initialized. 
        # If we deploy with gunicorn -w 1, it's fine.
        
        f_res = self.face_mesh.process(frame_rgb)
        h_res = self.hands.process(frame_rgb)
        
        live_face = self.get_face_features(f_res.multi_face_landmarks)
        
        live_hands_features = []
        if h_res.multi_hand_landmarks:
            for hand_landmarks in h_res.multi_hand_landmarks:
                feats = self.get_hand_features(hand_landmarks)
                if feats is not None:
                    live_hands_features.append(feats)
        
        if not live_hands_features:
            live_hands_features = [None]
            
        best_overall_match = None
        best_overall_score = -1.0
        
        for hand_feat in live_hands_features:
            m_name, m_score = self.find_best_match(live_face, hand_feat)
            if m_score > best_overall_score:
                best_overall_score = m_score
                best_overall_match = m_name
                
        return best_overall_match, best_overall_score
