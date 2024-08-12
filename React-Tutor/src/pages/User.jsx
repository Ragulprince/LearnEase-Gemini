import React, { useEffect, useState } from 'react';
import './User.css';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS
import { Button } from 'react-bootstrap'; // Import Bootstrap Button component

function User() {
  const [user, setUser] = useState(null);

  const handleGoClick = () => {
    // Validate URL before redirecting
    const url = 'http://localhost:8501';
    if (url) {
      window.location.href = url; // URL of your Streamlit app
    } else {
      console.error('Invalid URL');
    }
  };

  useEffect(() => {
    // Fetch user details from localStorage
    const userDetails = JSON.parse(localStorage.getItem('user'));
    if (userDetails) {
      console.log("User details fetched:", userDetails); // Debugging statement
      setUser(userDetails);
    } else {
      console.log("No user details found in localStorage."); // Debugging statement
    }
  }, []);

  return (
    <div className="welcome word">
      <span id="" className="splash"></span>
      <span id="welcome" className="z-depth-4"></span>
      
      <header className="navbar-fixed">
        <nav className="row deep-purple darken-3">
          <div className="col s12">
            <ul className="right">
              {/* Add navigation items here if needed */}
            </ul>
          </div>
        </nav>
      </header>

      <main className="valign-wrapper">
        <span className="container grey-text text-lighten-1">
          <p className="flow-text inv">Welcome Back</p>
          
          {/* User Details Section */}
          <div className="user-details center-align">
            {user ? (
              <>
                {user.photoURL ? (
                  <img src={user.photoURL} alt={user.name} className="user-photo" />
                ) : (
                  <img src="default-avatar.png" alt="Default Avatar" className="user-photo" />
                )}
                <h2 className="user-name">{user.name}</h2>
              </>
            ) : (
              <>
                <img src="default-avatar.png" alt="Default Avatar" className="user-photo" />
                <h1 className="title grey-text text-lighten-3">User Name</h1>
              </>
            )}
          </div>
          
          <blockquote className="yyy">
            Elevate your education with an AI tutor that's as dynamic as you are
          </blockquote>

          <div className="center-align">
          <Button onClick={handleGoClick} variant="primary">Dive In</Button>

          </div>
        </span>
      </main>
    </div>
  );
}

export default User;
