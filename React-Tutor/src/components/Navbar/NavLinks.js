import React from 'react';
import { HashLink } from 'react-router-hash-link';
import { signInWithGoogle } from '../../pages/Firebase';

const NavLinks = () => {
    const handleSignIn = async () => {
        try {
          await signInWithGoogle();
        } catch (error) {
          console.error('Error during sign-in:', error);
        }
      };
    return (
        <>
            {/* <HashLink className="text-white bg-blue-900 hover:bg-blue-800 inline-flex items-center justify-center w-auto px-6 py-3 shadow-xl rounded-xl" smooth to="/second">
                Get started
            </HashLink> */}
            <button className="text-white bg-blue-900 hover:bg-blue-800 inline-flex items-center justify-center w-auto px-6 py-3 shadow-xl rounded-xl" onClick={handleSignIn} >Get started</button>
        </>
    )
}

export default NavLinks;
