import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Game from './pages/Game';
import Stats from './pages/Stats';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-b from-purple-100 to-pink-100">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/game" element={<Game />} />
          <Route path="/stats" element={<Stats />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App; 