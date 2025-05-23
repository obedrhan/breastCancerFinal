import React, { useState } from 'react';
import './LoginPage.css';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';

const BASE_URL = 'http://127.0.0.1:5000';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await fetch(`${BASE_URL}/login`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (res.ok) {
        toast.success("Login successful");
        navigate('/home');
      } else {
        const msg = await res.text();
        toast.error(msg || "Login failed");
      }
    } catch (err) {
      console.error("❌ Login error:", err);
      toast.error("Could not connect to the server.");
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit} className="login-form">
        <h1 className="login-title">Breast Cancer Detection System</h1>

        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
        />
        <button type="submit">Log In</button>

        <p className="register-link">
          Don’t have an account? <Link to="/register">Click here to register</Link>
        </p>
      </form>
    </div>
  );
}

export default LoginPage;