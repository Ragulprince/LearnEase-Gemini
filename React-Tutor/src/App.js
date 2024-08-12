import React, { useEffect } from 'react';
import AOS from 'aos';
import "aos/dist/aos.css";
import Second from './pages/Second';
import {
  BrowserRouter as Router,
  Routes,
  Route
} from 'react-router-dom';
// All pages
import Home from './pages/Home';

import {useDocTitle} from './components/CustomHook';
import ScrollToTop from './components/ScrollToTop';

import User from './pages/User';
function App() {
  useEffect(() => {
    const aos_init = () => {
      AOS.init({
        once: true,
        duration: 1000,
        easing: 'ease-out-cubic',
      });
    }

    window.addEventListener('load', () => {
      aos_init();
    });
  }, []);

  useDocTitle("LearnEase");

  return (
    <>
      <Router>
        <ScrollToTop>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/second" element={<User />} /> 
          </Routes>
        </ScrollToTop>
      </Router>
    </>
  );
}


export default App;
