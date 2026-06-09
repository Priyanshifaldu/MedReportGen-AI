// frontend/src/components/UploadForm.jsx
import React, { useState } from 'react';

const UploadForm = ({ onGenerate, loading }) => {
  const [formData, setFormData] = useState({
    age: 30, // Default
    gender: 'Male', // Default
    pain_intensity: 5,
    hemoglobin: 8.5,
    oxygen_saturation: 95.0,
    pain_type: 'Legs',
    facility_type: 'ER',
    location: 'Bronx',
    admitted: 'No'
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    // Handle number inputs
    const parsedValue = name.includes('intensity') || name.includes('saturation') || name.includes('hemoglobin') ? parseFloat(value) : value;
    setFormData(prev => ({ ...prev, [name]: parsedValue }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Structure the payload according to GenerationRequest schema
    const payload = {
      summary: { // <-- Wrap the data in a 'summary' object
        age: parseInt(formData.age),
        gender: formData.gender,
        pain_intensity: parseInt(formData.pain_intensity),
        hemoglobin: parseFloat(formData.hemoglobin),
        oxygen_saturation: parseFloat(formData.oxygen_saturation),
        pain_type: formData.pain_type,
        facility_type: formData.facility_type,
        location: formData.location,
        admitted: formData.admitted
      }
    };

    onGenerate(payload); // Pass the correctly structured payload
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow-md">
      <h2 className="text-xl font-bold text-gray-800 mb-4">Patient Summary Input</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
            <input
              name="age"
              type="number"
              value={formData.age}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="0"
              max="150"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
            <select
              name="gender"
              value={formData.gender}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option>Male</option>
              <option>Female</option>
              <option>Other</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Pain Intensity (1-10)</label>
            <input
              name="pain_intensity"
              type="number"
              value={formData.pain_intensity}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="10"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Hemoglobin (g/dL)</label>
            <input
              name="hemoglobin"
              type="number"
              step="0.1"
              value={formData.hemoglobin}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="0"
              max="20"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Oxygen Saturation (%)</label>
            <input
              name="oxygen_saturation"
              type="number"
              step="0.1"
              value={formData.oxygen_saturation}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="0"
              max="100"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Facility Type</label>
            <select
              name="facility_type"
              value={formData.facility_type}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option>ER</option>
              <option>Urgent Care</option>
              <option>Outpatient</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Pain Type</label>
            <select
              name="pain_type"
              value={formData.pain_type}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option>Legs</option>
              <option>Chest</option>
              <option>Abdomen</option>
              <option>Back</option>
              <option>Head</option>
              <option>Arms</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Location</label>
            <input
              name="location"
              type="text"
              value={formData.location}
              onChange={handleChange}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              placeholder="e.g., Bronx"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Admitted?</label>
          <select
            name="admitted"
            value={formData.admitted}
            onChange={handleChange}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          >
            <option>No</option>
            <option>Yes</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className={`w-full py-2 px-4 rounded-md text-white font-medium ${
            loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {loading ? 'Generating...' : 'Generate Clinical Note'}
        </button>
      </form>
    </div>
  );
};

export default UploadForm;