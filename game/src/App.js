import React, { useState } from 'react';
import './App.css'; // You can create this file for styling

const emails = [
  {
    subject: "Urgent: Your account is suspended!",
    body: "Please click this link to verify your account immediately or it will be deleted.",
    isPhishing: true,
  },
  {
    subject: "Your order is on its way!",
    body: "Hi [Name], your Amazon order has shipped. Track it here: [valid link]",
    isPhishing: false,
  },
  {
    subject: "You've won a free iPhone!",
    body: "Congratulations! You have been selected as our winner. Claim your prize now!",
    isPhishing: true,
  },
];

function App() {
  const [currentEmailIndex, setCurrentEmailIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [feedback, setFeedback] = useState('');

  const handleGuess = (isGuessPhishing) => {
    const isCorrect = isGuessPhishing === emails[currentEmailIndex].isPhishing;
    if (isCorrect) {
      setScore(score + 1);
      setFeedback('Correct!');
    } else {
      setFeedback('Incorrect.');
    }

    setTimeout(() => {
      setFeedback('');
      if (currentEmailIndex < emails.length - 1) {
        setCurrentEmailIndex(currentEmailIndex + 1);
      } else {
        alert(`Game Over! Your final score is: ${score}/${emails.length}`);
        setCurrentEmailIndex(0);
        setScore(0);
      }
    }, 1500);
  };

  const currentEmail = emails[currentEmailIndex];

  return (
    <div className="App">
      <div className="game-container">
        <h1>Phishing Game</h1>
        <p className="score">Score: {score}</p>
        <div className="email-box">
          <div className="email-header">
            <h3>Subject: {currentEmail.subject}</h3>
          </div>
          <div className="email-body">
            <p>{currentEmail.body}</p>
          </div>
        </div>

        <div className="button-container">
          <button onClick={() => handleGuess(true)}>This is a Phish</button>
          <button onClick={() => handleGuess(false)}>This is Legitimate</button>
        </div>
        
        <p className="feedback">{feedback}</p>
      </div>
    </div>
  );
}

export default App;