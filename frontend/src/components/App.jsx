// frontend/src/App.jsx
import React, { useState } from 'react';
import UploadForm from './components/UploadForm';
import ReportOutput from './components/ReportOutput';

function App() {
  const [note, setNote] = useState('');
  const [warnings, setWarnings] = useState([]);
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async (payload) => {
    setLoading(true);
    setError('');
    try {
      // Use a RELATIVE path for the API call
      // This means the request will go to http://localhost:8000/generate (if backend runs on 8000)
      // because the frontend is served from http://localhost:8000/
      const response = await fetch('/generate', { // <-- Changed from 'http://localhost:8000/generate'
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate note');
      }

      const data = await response.json();
      setNote(data.generated_note);
      setWarnings(data.warnings);
      setConfidence(data.confidence_score);
    } catch (err) {
      console.error("API Error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8">
      {/* Header */}
      <header className="mb-8 flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <div className="bg-blue-600 w-10 h-10 rounded-lg flex items-center justify-center text-white font-bold">V</div>
          <div>
            <h1 className="text-2xl font-bold text-gray-800">MedReportGen AI</h1>
            <p className="text-sm text-gray-600">AI-Powered Medical Reporting</p>
          </div>
        </div>
        <nav className="hidden md:flex space-x-6">
          <a href="#" className="text-gray-800 hover:text-blue-600 font-medium">Dashboard</a>
          <a href="#" className="text-blue-600 font-medium border-b-2 border-blue-600">Generate Report</a>
          <a href="#" className="text-gray-800 hover:text-blue-600 font-medium">Saved Reports</a>
          <a href="#" className="text-gray-800 hover:text-blue-600 font-medium">Explainability View</a>
          <a href="#" className="text-gray-800 hover:text-blue-600 font-medium">Settings</a>
        </nav>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">Dr. Sarah Chen</span>
          <div className="bg-blue-600 w-10 h-10 rounded-full flex items-center justify-center text-white font-bold">SC</div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left Panel: Form and Output */}
          <div className="lg:col-span-3 space-y-6">
            <UploadForm onGenerate={handleGenerate} loading={loading} />
            {loading && <div className="text-center text-blue-600">Generating...</div>}
            {error && <div className="text-center text-red-600 p-4 bg-red-50 rounded-lg">Error: {error}</div>}
            {note && <ReportOutput note={note} warnings={warnings} confidence={confidence} />}
          </div>

          {/* Right Panel: Metrics Dashboard */}
          <div className="lg:col-span-1">
            <div className="bg-white p-6 rounded-xl shadow-md h-fit">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Dashboard Metrics</h2>
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">Notes Generated</p>
                  <p className="text-2xl font-bold text-blue-600">0</p> {/* Placeholder */}
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="text-sm text-green-800">Avg. Confidence</p>
                  <p className="text-2xl font-bold text-green-600">{confidence ? `${(confidence * 100).toFixed(1)}%` : 'N/A'}</p>
                </div>
                <div className="p-4 bg-yellow-50 rounded-lg">
                  <p className="text-sm text-yellow-800">Warnings Raised</p>
                  <p className="text-2xl font-bold text-yellow-600">{warnings.length}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;