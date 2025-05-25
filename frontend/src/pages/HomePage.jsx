import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import './HomePage.css';

const BASE_URL = 'http://127.0.0.1:5000';

function HomePage() {
  const [file, setFile] = useState(null);
  const [patientName, setPatientName] = useState('');
  const [nationalId, setNationalId] = useState('');
  const [result, setResult] = useState(null);
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      const res = await fetch(`${BASE_URL}/logout`, {
        method: 'POST',
        credentials: 'include'
      });
      if (res.ok) {
        toast.success("Logged out");
        navigate('/');
      } else {
        toast.error("Logout failed");
      }
    } catch (err) {
      toast.error("Logout error");
      console.error("Logout error:", err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_name', patientName);
    formData.append('national_id', nationalId);

    try {
      const res = await fetch(`${BASE_URL}/process`, {
        method: 'POST',
        credentials: 'include',
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        setResult(data);
        toast.success("Prediction successful!");
      } else {
        const errText = await res.text();
        console.error("❌ Upload failed:", errText);
        toast.error(errText || "Error processing request");
      }
    } catch (err) {
      console.error("❌ Network error:", err);
      toast.error("Server error");
    }
  };

  return (
    <div className="home-page">
      <header className="home-header">
        <h1 className="header-title">Breast Cancer Detection System</h1>
        <div className="header-buttons">
          <button onClick={() => navigate('/results')} className="header-btn secondary">View Previous Results</button>
          <button onClick={handleLogout} className="header-btn danger">Logout</button>
        </div>
      </header>

      <form className="upload-form" onSubmit={handleSubmit}>
        <h2>Upload Mammogram</h2>
        <input type="text" placeholder="Patient Full Name" value={patientName} onChange={e => setPatientName(e.target.value)} required />
        <input type="text" placeholder="National ID" value={nationalId} onChange={e => setNationalId(e.target.value)} required />
        <input type="file" accept=".dcm" onChange={e => setFile(e.target.files[0])} required />
        <button type="submit">Submit</button>
      </form>

      {result && (
        <div className="result-box">
          <h3>Prediction Result</h3>
          <p><strong>Prediction:</strong> {result.prediction.toUpperCase()}</p>
          <p><strong>Confidence:</strong> {result.confidence}%</p>
          <img src={result.image_url} alt="Segmented Output" className="segmented-image" />
        </div>
      )}
    </div>
  );
}

export default HomePage;