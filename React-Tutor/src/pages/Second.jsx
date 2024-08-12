// // src/components/Second.jsx
// import React, { useEffect, useState } from 'react';
// import './Second.css';

// function Second() {
//   const [user, setUser] = useState(null);

//   useEffect(() => {
//     const userDetails = JSON.parse(localStorage.getItem('user'));
//     if (userDetails) {
//       setUser(userDetails);
//     }
//   }, []);

//   const handleGoClick = () => {
//     window.location.href = 'http://localhost:8501'; // URL of your Streamlit app
//   };

//   return (
//     <div className="full1">
//       <span id="splash-overlay" className="splash"></span>
//       <span id="welcome" className="z-depth-4"></span>
      
//       <header className="navbar-fixed">
//         <nav className="row deep-purple darken-3">
//           <div className="col s12">
//             <ul className="right">
//               {/* Add navigation items here if needed */}
//             </ul>
//           </div>
//         </nav>
//       </header>
        
//       <p className="flow-text inv">Welcome Back</p>
//       <blockquote className="yyy">
//             Elevate your education with an AI tutor that's as dynamic as you are
//         </blockquote>
        
//       <div className="second-card">
//         <div className="second-header">
//           {user && user.photoURL ? (
//             <img src={user.photoURL} alt={user.name} className="second-img" />
//           ) : (
//             <img src="default-avatar.png" alt="Default Avatar" className="second-img" />
//           )}
//           <div className="second-details ">
//             <span className="second-name outer">{user ? user.name : 'Name'}</span>
//           </div>
//         </div>
//         <div className="second-btns">
//           <div className="second-btn second-btn-2 second-shimmer" onClick={handleGoClick}>
//             Dive IN 
//           </div>
//         </div>
//       </div>
//       </div>
//   );
// }

// export default Second;
