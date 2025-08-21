import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const emotions = [
  { name: 'angry', label: 'Be angry!', emoji: '😠' },
  { name: 'disgusted', label: 'Be disgusted!', emoji: '🤢' },
  { name: 'fearful', label: 'Be fearful!', emoji: '😱' },
  { name: 'happy', label: 'Be happy!', emoji: '😃' },
  { name: 'sad', label: 'Be sad!', emoji: '😢' },
  { name: 'surprised', label: 'Be surprised!', emoji: '😲' },
  { name: 'neutral', label: 'Be neutral!', emoji: '😐' },
];

// Function to get emoji for emotion
function getEmojiForEmotion(emotionName) {
  const found = emotions.find(e => e.name === emotionName);
  return found ? found.emoji : '❓';
}

function getEmotionInfo(emotionName) {
  switch (emotionName) {
    case 'angry':
      return {
        title: 'Understanding Anger',
        description: 'Anger is a strong feeling of displeasure or hostility. It can be triggered by frustration, injustice, or feeling threatened.',
        signs: [
          'Furrowed brows',
          'Tightened jaw',
          'Flared nostrils',
          'Tense facial muscles',
        ],
        feels: 'You may feel hot, tense, or have a strong urge to act.',
        tips: 'Take a deep breath and focus on what made you angry. Let the feeling show on your face.'
      };
    case 'disgusted':
      return {
        title: 'Understanding Disgust',
        description: 'Disgust is a feeling of revulsion or profound disapproval aroused by something unpleasant or offensive.',
        signs: [
          'Wrinkled nose',
          'Raised upper lip',
          'Narrowed eyes',
          'Head pulled back',
        ],
        feels: 'You may feel a strong urge to avoid or reject something.',
        tips: 'Think of something you find gross or unpleasant and let your face react.'
      };
    case 'fearful':
      return {
        title: 'Understanding Fear',
        description: 'Fear is an emotional response to a perceived threat or danger.',
        signs: [
          'Wide eyes',
          'Raised eyebrows',
          'Mouth slightly open',
          'Tense facial muscles',
        ],
        feels: 'You may feel tense, alert, or want to escape.',
        tips: 'Imagine something scary and let your face show your reaction.'
      };
    case 'happy':
      return {
        title: 'Understanding Happiness',
        description: 'Happiness is a positive emotion characterized by feelings of joy, satisfaction, contentment, and fulfillment.',
        signs: [
          'Smiling with raised corners of the mouth',
          'Crinkled eyes ("crow\'s feet")',
          'Raised cheeks',
          'Relaxed facial muscles',
        ],
        feels: 'You may feel light, energetic, and optimistic. The body may feel relaxed yet energized.',
        tips: 'Think of something that brings you joy and let that feeling spread to your face.'
      };
    case 'sad':
      return {
        title: 'Understanding Sadness',
        description: 'Sadness is an emotional pain associated with, or characterized by, feelings of disadvantage, loss, despair, grief, helplessness, disappointment, and sorrow.',
        signs: [
          'Downturned mouth',
          'Drooping eyelids',
          'Lowered eyebrows',
          'Tearful eyes',
        ],
        feels: 'You may feel heavy, slow, or want to withdraw.',
        tips: 'Think of a sad memory or loss and let your face reflect that emotion.'
      };
    case 'surprised':
      return {
        title: 'Understanding Surprise',
        description: 'Surprise is a brief emotional state experienced as the result of an unexpected event.',
        signs: [
          'Raised eyebrows',
          'Wide open eyes',
          'Mouth open in an O shape',
          'Relaxed jaw',
        ],
        feels: 'You may feel a jolt, alertness, or curiosity.',
        tips: 'Imagine something unexpected just happened and let your face react.'
      };
    case 'neutral':
      return {
        title: 'Understanding Neutral',
        description: 'A neutral expression shows no strong emotion and is often used as a baseline for comparison.',
        signs: [
          'Relaxed mouth',
          'No tension in the face',
          'Natural eye position',
          'No raised eyebrows',
        ],
        feels: 'You may feel calm, balanced, or indifferent.',
        tips: 'Relax your face and let it settle into its natural state.'
      };
    default:
      return null;
  }
}

function Game() {
  const [level, setLevel] = useState(1);
  const [current, setCurrent] = useState(() => Math.floor(Math.random() * emotions.length));
  const [stars, setStars] = useState(0);
  const [feedback, setFeedback] = useState('');
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [faceBox, setFaceBox] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [isCorrect, setIsCorrect] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);

  const VIDEO_WIDTH = 480;
  const VIDEO_HEIGHT = 360;

  const navigate = useNavigate();

  // Load face detector
  useEffect(() => {
    async function loadDetector() {
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      const detector = await faceLandmarksDetection.createDetector(model, {
        runtime: 'tfjs',
        refineLandmarks: true,
        maxFaces: 1,
      });
      detectorRef.current = detector;
    }
    loadDetector();
  }, []);

  // Start camera
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;
    setIsCameraOn(true);
  };

  // Detect and crop face, then send to backend
  const checkEmotion = async () => {
    setLoading(true);
    setFeedback('');
    setPrediction(null);
    setFaceBox(null);
    setShowModal(false);
    setIsCorrect(null);
    const detector = detectorRef.current;
    if (!detector || !videoRef.current) {
      setFeedback('Face detector or camera not ready.');
      setLoading(false);
      return;
    }

    // Draw the full video frame to canvas (no cropping)
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const base64 = canvas.toDataURL('image/png');

    // Optionally, you can still run face detection in the frontend for overlay, but do not crop for backend
    const faces = await detector.estimateFaces(videoRef.current);
    console.log('Face detection result:', faces);
    if (faces.length) {
      const box = faces[0].box;
      const [x, y, width, height] = [
        Math.max(0, box.xMin),
        Math.max(0, box.yMin),
        box.xMax - box.xMin,
        box.yMax - box.yMin,
      ];
      // Add 15% padding to each side
      const PADDING_X = width * 0.20;
      const PADDING_Y = height * 0.30;
      const paddedBox = {
        x: Math.max(0, x - PADDING_X),
        y: Math.max(0, y - PADDING_Y),
        width: Math.min(VIDEO_WIDTH, width + 2 * PADDING_X),
        height: Math.min(VIDEO_HEIGHT, height + 2 * PADDING_Y),
      };
      setFaceBox(paddedBox);
    } else {
      setFaceBox(null);
    }

    // Send the full frame to backend
    try {
      const res = await axios.post('http://localhost:5000/api/check-emotion', {
        image: base64,
        targetEmotion: emotions[current].name,
        userId: 'test_user'
      });
      setFeedback(res.data.feedback);
      setPrediction({
        emotion: res.data.predictedEmotion,
        confidence: res.data.confidence
      });
      setIsCorrect(res.data.isCorrect);
      setShowModal(true);
      if (res.data.isCorrect) {
        setStars(stars + 1);
        if ((stars + 1) % 3 === 0) setLevel(level + 1);
      }
    } catch (err) {
      setFeedback('Error: ' + (err.response?.data?.error || err.message));
      setIsCorrect(false);
      setShowModal(true);
    }
    setLoading(false);
  };

  const emotion = emotions[current];
  const emotionInfo = getEmotionInfo(emotion.name);

  const nextEmotion = () => {
    let newIndex;
    do {
      newIndex = Math.floor(Math.random() * emotions.length);
    } while (newIndex === current && emotions.length > 1);
    setCurrent(newIndex);
    setFeedback('');
    setPrediction(null);
    setShowModal(false);
    setFaceBox(null);
    setStars(0);
  };

  const retryEmotion = () => {
    setFeedback('');
    setPrediction(null);
    setShowModal(false);
    setFaceBox(null);
  };

  return (
    <>
      {/* Modal Popup for Results */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-md w-full text-center relative animate-fade-in">
            <button
              className="absolute top-2 right-4 text-2xl text-purple-400 hover:text-purple-600 font-bold"
              onClick={() => setShowModal(false)}
              aria-label="Close"
            >
              ×
            </button>
            <p className="text-2xl font-bold mb-2 text-purple-600">Result</p>
            <p className="text-xl mb-2">
              {feedback.startsWith('Error:') ? (
                <span><span style={{ color: 'red', fontWeight: 'bold' }}>Error:</span>{feedback.slice(6)}</span>
              ) : (
                feedback
              )}
            </p>
            {prediction && (
              <div className="mb-2">
                <span className="text-gray-600">Model prediction: </span>
                <span className="font-bold text-purple-600 text-lg">
                  {prediction.emotion} {getEmojiForEmotion(prediction.emotion)}
                </span>
                <br />
                <span className="text-gray-500 text-sm">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mt-4">
              <button
                className="bg-purple-400 hover:bg-purple-500 text-white font-bold py-2 px-6 rounded-full text-lg shadow-md transition duration-200"
                onClick={retryEmotion}
              >
                Retry This Emotion
              </button>
              <button
                className={`${isCorrect === false ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'} text-white font-bold py-2 px-6 rounded-full text-lg shadow-md transition duration-200`}
                onClick={nextEmotion}
              >
                Next Emotion
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Info Modal */}
      {showInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-2xl shadow-2xl p-8 max-w-lg w-full text-left relative animate-fade-in">
            <button
              className="absolute top-2 right-4 text-2xl text-purple-400 hover:text-purple-600 font-bold"
              onClick={() => setShowInfo(false)}
              aria-label="Close"
            >
              ×
            </button>
            <h2 className="text-2xl font-bold mb-2 text-gray-800">{emotionInfo.title}</h2>
            <p className="mb-4 text-gray-700">{emotionInfo.description}</p>
            <div className="mb-2">
              <span className="font-semibold text-gray-800">Physical Signs:</span>
              <ul className="list-disc ml-6 text-gray-700">
                {emotionInfo.signs.map((sign, idx) => (
                  <li key={idx}>{sign}</li>
                ))}
              </ul>
            </div>
            <div className="mb-2">
              <span className="font-semibold text-gray-800">How It Feels:</span>
              <p className="ml-2 text-gray-700">{emotionInfo.feels}</p>
            </div>
            <div>
              <span className="font-semibold text-gray-800">Tips for Expressing:</span>
              <p className="ml-2 text-gray-700">{emotionInfo.tips}</p>
            </div>
          </div>
        </div>
      )}
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-400 to-blue-400">
        <div className="bg-gray-50 rounded-3xl shadow-2xl p-8 w-full max-w-4xl mx-4 flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <button
              className="text-purple-500 text-2xl font-bold flex items-center hover:underline focus:outline-none"
              onClick={() => navigate('/')}
              style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer' }}
            >
              <span className="mr-2">&larr;</span> Main Menu
            </button>
          </div>
          <div className="flex flex-col items-center mb-4">
            <h2 className="text-3xl md:text-4xl font-extrabold text-purple-600 mb-2 text-center">
              {emotion.label}
            </h2>
            <div className="text-5xl mb-2">{emotion.emoji}</div>
            <div className="flex items-center justify-center mb-2">
              {[...Array(3)].map((_, i) => (
                <span key={i} className={`text-2xl mx-1 ${i < stars % 3 ? 'text-yellow-400' : 'text-gray-300'}`}>★</span>
              ))}
            </div>
            <div className="w-full max-w-xl h-72 bg-gray-100 rounded-2xl shadow mt-2 flex items-center justify-center relative">            <video
                ref={videoRef}
                autoPlay
                playsInline
                width={VIDEO_WIDTH}
                height={VIDEO_HEIGHT}
                className="absolute w-full h-full object-cover rounded-2xl"
                style={{ display: isCameraOn ? 'block' : 'none' }}
              />
              {isCameraOn && faceBox && (
                <div
                  style={{
                    position: 'absolute',
                    border: '4px solid #a78bfa',
                    boxShadow: '0 0 16px 4px #a78bfa88',
                    left: `${(faceBox.x / VIDEO_WIDTH) * 100}%`,
                    top: `${(faceBox.y / VIDEO_HEIGHT) * 100}%`,
                    width: `${(faceBox.width / VIDEO_WIDTH) * 100}%`,
                    height: `${(faceBox.height / VIDEO_HEIGHT) * 100}%`,
                    boxSizing: 'border-box',
                    pointerEvents: 'none',
                    borderRadius: '16px',
                    background: 'rgba(167, 139, 250, 0.05)',
                  }}
                />
              )}
              <canvas
                ref={canvasRef}
                width={VIDEO_WIDTH}
                height={VIDEO_HEIGHT}
                style={{ display: 'none' }}
              />
              {!isCameraOn && (
                <span className="text-gray-400">Camera is off</span>
              )}
            </div>
          </div>
          <div className="flex justify-center mt-6 gap-4">
            {!isCameraOn ? (
              <button
                onClick={startCamera}
                className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-10 rounded-full text-lg shadow-md transition duration-200"
              >
                Start
              </button>
            ) : (
              <>
                <button
                  onClick={checkEmotion}
                  className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-10 rounded-full text-lg shadow-md transition duration-200"
                  disabled={loading}
                >
                  {loading ? 'Checking...' : 'Check Expression'}
                </button>
                <button
                  onClick={() => setShowInfo(true)}
                  className="ml-4 border-2 border-purple-500 text-purple-600 font-bold py-2 px-8 rounded-full text-lg transition duration-200 hover:bg-purple-50"
                >
                  Learn More
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default Game; 