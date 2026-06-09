// frontend/src/components/ReportOutput.jsx
import React from 'react';

const ReportOutput = ({ note, warnings, confidence }) => {
  return (
    <div className="bg-white p-6 rounded-xl shadow-md">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-xl font-bold text-gray-800">Generated Medical Report (AI Draft)</h2>
          <p className="text-sm text-gray-600">AI-generated report ready for review</p>
        </div>
        <div className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-medium">Generated</div>
      </div>

      <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 mb-4 max-h-96 overflow-y-auto">
        {note ? (
          <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">{note}</pre>
        ) : (
          <p className="text-gray-500 italic">Clinical note will appear here...</p>
        )}
      </div>

      <div className="flex flex-wrap justify-between items-center gap-2">
        <div className="text-sm">
          <span className="font-medium text-gray-700">Confidence:</span>{' '}
          <span className="font-bold">{confidence ? `${(confidence * 100).toFixed(1)}%` : 'N/A'}</span>
        </div>
        {warnings && warnings.length > 0 && (
          <div className="text-sm">
            <span className="font-medium text-red-700">Warnings:</span>{' '}
            <span className="text-red-600">{warnings.join(', ')}</span>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="mt-6 flex space-x-2">
        <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
          Regenerate Report
        </button>
        <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50">
          Edit Text
        </button>
        <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50">
          Explain Output
        </button>
        <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50">
          Copy to Clipboard
        </button>
      </div>

      {/* Quality Metrics */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Quality Metrics</span>
        </div>
        <div className="flex space-x-2">
          <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">BLEU: 0.87</span>
          <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium">ROUGE: 0.84</span>
          <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs font-medium">BERT: 0.91</span>
        </div>
      </div>

      {/* Footer Metrics */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <span className="text-sm text-gray-600">Reports Generated</span>
            <p className="text-xl font-bold text-gray-800">1,247</p>
          </div>
          <div className="text-center">
            <span className="text-sm text-gray-600">Approval Rate</span>
            <p className="text-xl font-bold text-gray-800">94.2%</p>
          </div>
          <div className="text-center">
            <span className="text-sm text-gray-600">Avg. Time Saved</span>
            <p className="text-xl font-bold text-gray-800">12.5 min</p>
          </div>
          <div className="text-center">
            <span className="text-sm text-gray-600">Avg. Report Length</span>
            <p className="text-xl font-bold text-gray-800">485 words</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportOutput;
