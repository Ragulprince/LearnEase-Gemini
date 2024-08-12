import { initializeApp } from "firebase/app";
import { getAuth, signInWithPopup, GoogleAuthProvider } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
    apiKey: "AIzaSyBEBcwUc3VKdY8o9CNsI_sS7TgefAlNZ24",
    authDomain: "aitutor-gemini.firebaseapp.com",
    projectId: "aitutor-gemini",
    storageBucket: "aitutor-gemini.appspot.com",
    messagingSenderId: "88454503935",
    appId: "1:88454503935:web:5b229db4709feab16d5bca",
    measurementId: "G-3MTYTKB7FW"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

export const signInWithGoogle = async () => {
  try {
    const result = await signInWithPopup(auth, provider);
    const user = result.user;
    
    // Store user details in localStorage
    localStorage.setItem('user', JSON.stringify({
      name: user.displayName,
      email: user.email,
      photoURL: user.photoURL,
    }));

    console.log('User signed in successfully');

    // Redirect to the second page
    window.location.href = "/Profile";

  } catch (error) {
    console.error('Error during sign-in:', error);
  }
};
