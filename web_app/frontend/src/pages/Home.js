import React from 'react';
import { useNavigate } from 'react-router-dom';

function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-purple-400 to-blue-400">
      <div className="bg-white rounded-3xl shadow-2xl p-10 w-full max-w-3xl mx-4">
        <div className="flex flex-col items-center mb-8">
          <h1 className="text-4xl font-extrabold text-purple-700 mb-2">MimicMe</h1>
          <h2 className="text-xl font-semibold text-gray-700 mb-2 text-center">
            Let’s make faces and have fun!
          </h2>
          <p className="text-gray-500 text-center mb-6 max-w-xl">
            Show your feelings with your face and get stars for doing great! Play, learn, and become an emotion expert!
          </p>
          <button
            onClick={() => navigate('/game')}
            className="bg-purple-500 hover:bg-purple-600 text-white font-bold py-3 px-8 rounded-full text-lg shadow-md transition duration-200"
          >
            Start Playing
          </button>
        </div>
        <div className="bg-purple-50 rounded-2xl p-6">
          <h3 className="text-lg font-bold text-gray-700 mb-4">How to Play</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white rounded-xl shadow p-4 flex flex-col items-center">
              <span className="text-2xl font-bold text-purple-500 mb-2">1</span>
              <span className="text-center text-gray-700">An emotion will appear!</span>
            </div>
            <div className="bg-white rounded-xl shadow p-4 flex flex-col items-center">
              <span className="text-2xl font-bold text-purple-500 mb-2">2</span>
              <span className="text-center text-gray-700">Make a face that matches the feeling</span>
            </div>
            <div className="bg-white rounded-xl shadow p-4 flex flex-col items-center">
              <span className="text-2xl font-bold text-purple-500 mb-2">3</span>
              <span className="text-center text-gray-700">Get feedback and earn stars!</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home; 