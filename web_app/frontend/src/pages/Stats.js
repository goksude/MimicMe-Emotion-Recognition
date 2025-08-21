import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D', '#FF6B6B'];

function Stats() {
  // Sample data - replace with actual data from backend
  const emotionData = [
    { name: 'Happy', value: 85 },
    { name: 'Sad', value: 65 },
    { name: 'Angry', value: 70 },
    { name: 'Surprised', value: 90 },
    { name: 'Neutral', value: 95 },
    { name: 'Disgust', value: 60 },
    { name: 'Fear', value: 55 },
  ];

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-purple-600 text-center mb-8">
        Your Progress
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Emotion Success Rate
          </h2>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={emotionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {emotionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Daily Achievements
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
              <span className="text-lg">Emotions Imitated Today</span>
              <span className="text-2xl font-bold text-purple-600">5</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
              <span className="text-lg">Current Streak</span>
              <span className="text-2xl font-bold text-purple-600">3 days</span>
            </div>
            <div className="flex items-center justify-between p-4 bg-purple-50 rounded-lg">
              <span className="text-lg">Total Stars Earned</span>
              <span className="text-2xl font-bold text-purple-600">15 ⭐</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">
          Areas for Improvement
        </h2>
        <div className="space-y-4">
          <div className="p-4 bg-yellow-50 rounded-lg">
            <h3 className="font-semibold text-yellow-800">Fear Expression</h3>
            <p className="text-gray-600">
              Try to express fear more naturally by widening your eyes and slightly opening your mouth.
            </p>
          </div>
          <div className="p-4 bg-yellow-50 rounded-lg">
            <h3 className="font-semibold text-yellow-800">Disgust Expression</h3>
            <p className="text-gray-600">
              Practice wrinkling your nose and raising your upper lip to show disgust more clearly.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Stats; 