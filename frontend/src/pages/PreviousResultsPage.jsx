import React, { useState } from 'react';
import './PreviousResultsPage.css';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';

const BASE_URL = 'http://127.0.0.1:5000';

function PreviousResultsPage() {
  const [searchName, setSearchName] = useState('');
  const [results, setResults] = useState([]);
  const navigate = useNavigate();

  const handleSearch = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch(`${BASE_URL}/view_previous`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_name: searchName })
      });

      if (res.ok) {
        const data = await res.json();
        setResults(data);
      } else {
        const errorMsg = await res.text();
        console.error("❌ Failed response:", errorMsg);
        toast.error("Failed to retrieve results.");
      }
    } catch (err) {
      console.error("❌ Network error:", err);
      toast.error("Error connecting to server.");
    }
  };

  return (
    <div className="results-container">
      <h2>Previous Classification Results</h2>

      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          placeholder="Search by patient name"
          value={searchName}
          onChange={(e) => setSearchName(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>

      <div className="results-list">
        {results.length === 0 ? (
          <p>No results yet.</p>
        ) : (
          results.map((r, i) => (
            <div className="result-card" key={i}>
              <p><strong>Patient:</strong> {r.fullname}</p>
              <p><strong>Prediction:</strong> {r.prediction}</p>
              <p><strong>Confidence:</strong> {r.confidence}%</p>
              <p><strong>Date:</strong> {r.date}</p>
            </div>
          ))
        )}
      </div>

      <div className="go-back">
        <button onClick={() => navigate('/home')} className="back-button">
          ← Back to Home
        </button>
      </div>
    </div>
  );
}

export default PreviousResultsPage;