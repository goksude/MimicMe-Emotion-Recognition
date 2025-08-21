from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the emotion recognition model
model = tf.keras.models.load_model('../emotion_detection_best3.keras')

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# In-memory storage for user statistics
user_stats = {}
daily_stats = {}

# Emotion labels
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def preprocess_image(image_data):
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None  # No face detected

    # Use the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_img = gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0

    # Save the preprocessed image for debugging
    cv2.imwrite('debug_face.png', (face_img * 255).astype('uint8'))

    # Reshape for model input
    face_img = np.expand_dims(face_img, axis=[0, -1])
    return face_img

@app.route('/api/check-emotion', methods=['POST'])
def check_emotion():
    data = request.json
    image_data = data.get('image')
    target_emotion = data.get('targetEmotion')
    user_id = data.get('userId', 'default_user')
    
    if not image_data or not target_emotion:
        return jsonify({'error': 'Missing image or target emotion'}), 400
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': "We couldn't find your face in the image. Please make sure your face is clearly visible to the camera and try again! 😊"}), 400
        
        # Get prediction
        predictions = model.predict(processed_image)
        print('Predictions:', predictions)
        predicted_emotion = EMOTIONS[np.argmax(predictions[0])]
        print('Predicted label:', predicted_emotion)
        confidence = float(np.max(predictions[0]))
        
        # Check if prediction matches target emotion
        is_correct = predicted_emotion == target_emotion
        
        # Update user stats
        update_user_stats(user_id, target_emotion, is_correct)
        
        return jsonify({
            'isCorrect': is_correct,
            'predictedEmotion': predicted_emotion,
            'confidence': confidence,
            'feedback': get_feedback(is_correct, confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_user_stats(user_id, emotion, is_correct):
    # Initialize user stats if not exists
    if user_id not in user_stats:
        user_stats[user_id] = {}
    
    if emotion not in user_stats[user_id]:
        user_stats[user_id][emotion] = {'attempts': 0, 'correct': 0}
    
    # Update emotion stats
    user_stats[user_id][emotion]['attempts'] += 1
    if is_correct:
        user_stats[user_id][emotion]['correct'] += 1
    
    # Update daily stats
    today = datetime.now().date().isoformat()
    if user_id not in daily_stats:
        daily_stats[user_id] = {}
    
    if today not in daily_stats[user_id]:
        daily_stats[user_id][today] = {'attempts': 0, 'correct': 0}
    
    daily_stats[user_id][today]['attempts'] += 1
    if is_correct:
        daily_stats[user_id][today]['correct'] += 1

def get_feedback(is_correct, confidence):
    if is_correct:
        if confidence > 0.8:
            return "Perfect! You nailed that expression! 🎉"
        elif confidence > 0.6:
            return "Great job! That's exactly right! 👏"
        else:
            return "Good! You got it right! 🌟"
    else:
        if confidence > 0.6:
            return "Almost there! Try to adjust your expression a bit more 💪"
        else:
            return "Not quite there yet. Let's try again! 🔄"

@app.route('/api/stats/<user_id>', methods=['GET'])
def get_user_stats(user_id):
    try:
        # Get emotion stats
        emotion_stats = {}
        if user_id in user_stats:
            for emotion, stats in user_stats[user_id].items():
                success_rate = (stats['correct'] / stats['attempts'] * 100) if stats['attempts'] > 0 else 0
                emotion_stats[emotion] = {
                    'attempts': stats['attempts'],
                    'correct': stats['correct'],
                    'success_rate': success_rate
                }
        
        # Get today's daily stats
        today = datetime.now().date().isoformat()
        daily_stat = {'attempts': 0, 'correct': 0}
        if user_id in daily_stats and today in daily_stats[user_id]:
            daily_stat = daily_stats[user_id][today]
        
        return jsonify({
            'emotionStats': emotion_stats,
            'dailyStats': daily_stat
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 