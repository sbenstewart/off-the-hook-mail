import React, { useState } from 'react';
import { AlertCircle, CheckCircle, XCircle, Mail, Shield, ArrowRight } from 'lucide-react';

const PhishingQuiz = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [quizStarted, setQuizStarted] = useState(false);
  const [results, setResults] = useState([]);

  const quizData = [
    {
      email: "Subject: URGENT: Verify your university payroll\nFrom: Payroll Desk <notice@finance-alerts-university.com>\nTo: Student <student@asu.edu>\n\nYour direct deposit will be paused unless you verify your bank in our new portal.\nUse the secure page: hxxps://finance-alerts-university.com/verify\nDo not share this email.",
      isPhishing: true,
      explanation: "This is a phishing attempt. Red flags include: (1) Suspicious sender domain 'finance-alerts-university.com' instead of official ASU domain, (2) Creates urgency with threats about pausing direct deposit, (3) Asks you to verify banking information via external link, (4) Uses 'hxxps' which is not a standard protocol.",
      redFlags: ["Suspicious sender domain", "Urgent threat", "Requests banking info", "External link"]
    },
    {
      email: "Subject: Faculty meeting moved to 3:15 PM\nFrom: Department Admin <admin@cs.asu.edu>\nTo: Faculty List <faculty@cs.asu.edu>\n\nQuick heads-up: today's faculty meeting is shifted to 3:15 PM in room 402.\nNo action needed if you can't make it. Minutes will be sent after.",
      isPhishing: false,
      explanation: "This is a legitimate email. Indicators include: (1) Sender uses official ASU domain (@cs.asu.edu), (2) No urgent threats or requests for personal information, (3) Reasonable content about schedule changes, (4) No suspicious links or attachments.",
      goodSigns: ["Official ASU domain", "No personal info requests", "Reasonable content", "No suspicious links"]
    },
    {
      email: "Subject: Final notice: storage quota exceeded\nFrom: Microsoft 365 <no-reply@micros0ft-support.com>\nTo: Student <student@asu.edu>\n\nWe could not deliver recent messages. Increase your mailbox quota within 24 hours.\nContinue here: http://micros0ft-support.com/quota\nFailure to act will result in mailbox deactivation.",
      isPhishing: true,
      explanation: "This is a phishing attempt. Red flags include: (1) Misspelled domain 'micros0ft-support.com' with a zero instead of 'o', (2) Creates false urgency with 24-hour deadline, (3) Threatens account deactivation, (4) ASU uses different email systems, not generic Microsoft notices.",
      redFlags: ["Misspelled domain (0 instead of o)", "False urgency", "Account threat", "Generic sender"]
    },
    {
      email: "Subject: Library workshop: Zotero 101\nFrom: University Library <library@asu.edu>\nTo: Students <students@asu.edu>\n\nJoin us for a 30-minute intro to Zotero reference management, Thursday at 4 pm.\nNo registration required. See you in the instruction lab.",
      isPhishing: false,
      explanation: "This is a legitimate email. Indicators include: (1) Official ASU library domain (@asu.edu), (2) Typical university service announcement, (3) No requests for personal information or credentials, (4) Reasonable educational content.",
      goodSigns: ["Official ASU domain", "Educational content", "No personal info requests", "Typical service"]
    },
    {
      email: "Subject: DocuSign: Action Required\nFrom: DocuSign Service <notify@docusign-services.net>\nTo: Student <student@asu.edu>\n\nA confidential document has been shared with you. View and sign to avoid delays.\nOpen: https://docusign-services.net/doc/9142\nNote: for security, sign in with your email and password.",
      isPhishing: true,
      explanation: "This is a phishing attempt. Red flags include: (1) Generic sender from suspicious domain 'docusign-services.net', (2) Creates urgency to 'avoid delays', (3) Asks you to sign in with credentials on external site, (4) Real DocuSign uses their official domain and doesn't ask for passwords in emails.",
      redFlags: ["Suspicious domain", "Urgency tactics", "Requests credentials", "Generic message"]
    },
    {
      email: "Subject: Thanks for volunteering at orientation!\nFrom: Student Life <studentlife@asu.edu>\nTo: Alex <alex@asu.edu>\n\nWe appreciate your help. T-shirts are available for pickup in the union 10-2 pm.\nPhotos are posted on our site. Have a great week!",
      isPhishing: false,
      explanation: "This is a legitimate email. Indicators include: (1) Official ASU domain (@asu.edu), (2) Personalized content about campus activities, (3) No suspicious links or requests for sensitive data, (4) Friendly tone typical of student services.",
      goodSigns: ["Official ASU domain", "Personalized content", "Campus-specific info", "No data requests"]
    },
    {
      email: "Subject: Security Alert: unusual sign-in attempt\nFrom: Account Security <security@acc0unt-verify.net>\nTo: Student <student@asu.edu>\n\nWe detected a sign-in from a new device. If this was you, ignore this message.\nIf not, verify your account immediately: https://acc0unt-verify.net/login",
      isPhishing: true,
      explanation: "This is a phishing attempt. Red flags include: (1) Fake domain 'acc0unt-verify.net' with zero instead of 'o', (2) Generic security alert not from official ASU systems, (3) Pressures immediate action to 'verify account', (4) External verification link designed to steal credentials.",
      redFlags: ["Fake domain (0 instead of o)", "Generic alert", "Immediate action pressure", "Credential harvesting"]
    },
    {
      email: "Subject: Seminar reminder: Distributed Systems\nFrom: CS Events <events@cs.asu.edu>\nTo: Grad Students <grad@cs.asu.edu>\n\nReminder: Dr. Patel's seminar is Friday at 11 am, Room 245.\nCoffee and snacks provided. Slides will be uploaded afterward.",
      isPhishing: false,
      explanation: "This is a legitimate email. Indicators include: (1) Official ASU CS department domain (@cs.asu.edu), (2) Specific academic event details, (3) No requests for personal information or logins, (4) Standard departmental communication format.",
      goodSigns: ["Official department domain", "Specific event details", "No personal info requests", "Academic content"]
    }
  ];

  const handleAnswer = (answer) => {
    if (answered) return;
    
    setSelectedAnswer(answer);
    setAnswered(true);
    
    const isCorrect = answer === quizData[currentQuestion].isPhishing;
    if (isCorrect) {
      setScore(score + 1);
    }
    
    setResults([...results, {
      question: currentQuestion,
      correct: isCorrect,
      userAnswer: answer
    }]);
  };

  const nextQuestion = () => {
    if (currentQuestion < quizData.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setAnswered(false);
      setSelectedAnswer(null);
    } else {
      setShowResult(true);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestion(0);
    setScore(0);
    setShowResult(false);
    setAnswered(false);
    setSelectedAnswer(null);
    setQuizStarted(false);
    setResults([]);
  };

  const startQuiz = () => {
    setQuizStarted(true);
  };

  if (!quizStarted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 to-red-50 flex items-center justify-center p-4">
        <div className="max-w-2xl w-full bg-white rounded-2xl shadow-2xl p-8">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <img 
                src="/asu-logo.png" 
                alt="Arizona State University Logo" 
                className="h-20 object-contain"
              />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Phishing Awareness Training</h1>
            <p className="text-xl text-gray-600 mb-6">Information Security Education Program</p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8 text-left">
              <h2 className="font-semibold text-lg mb-3 flex items-center gap-2 text-gray-800">
                <Shield className="w-5 h-5 text-blue-600" />
                Training Overview
              </h2>
              <p className="text-gray-700 mb-4">
                This interactive training module will help you identify phishing attempts and protect your ASU credentials. 
                Phishing is one of the most common cyber threats facing universities today.
              </p>
              <ul className="space-y-2 text-gray-700">
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Test your ability to identify phishing emails</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Learn to spot common red flags and suspicious patterns</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>8 realistic email scenarios based on actual threats</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Receive detailed explanations after each answer</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1">•</span>
                  <span>Protect yourself and the ASU community</span>
                </li>
              </ul>
            </div>
            <button
              onClick={startQuiz}
              className="bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-900 hover:to-black text-white px-8 py-4 rounded-lg text-lg font-semibold transition-all transform hover:scale-105 flex items-center gap-2 mx-auto shadow-lg"
            >
              Begin Training
              <ArrowRight className="w-5 h-5" />
            </button>
            <p className="text-sm text-gray-500 mt-6">
              Estimated time: 10-15 minutes
            </p>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs text-gray-500">
                ASU Information Security | For assistance, contact security@asu.edu
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (showResult) {
    const percentage = Math.round((score / quizData.length) * 100);
    let message = "";
    let messageColor = "";
    
    if (percentage >= 80) {
      message = "Excellent! You're a phishing detection expert!";
      messageColor = "text-green-600";
    } else if (percentage >= 60) {
      message = "Good job! Keep practicing to improve your skills.";
      messageColor = "text-blue-600";
    } else {
      message = "Keep learning! Review the explanations to improve.";
      messageColor = "text-orange-600";
    }

    return (
      <div className="min-h-screen bg-gradient-to-br from-amber-50 to-red-50 p-4">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-6 mb-6 flex items-center justify-between">
            <img 
              src="/asu-logo.png"
              alt="Arizona State University" 
              className="h-12 object-contain"
            />
            <div className="text-right">
              <div className="text-sm text-gray-600">Training Complete</div>
              <div className="text-lg font-semibold text-gray-800">Final Results</div>
            </div>
          </div>
          
          <div className="bg-white rounded-2xl shadow-2xl p-8 mb-6">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-2">Training Complete</h2>
              <p className="text-xl text-gray-600 mb-6">Here are your results</p>
              
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-8 mb-6 border border-blue-200">
                <div className="text-6xl font-bold text-gray-800 mb-2">
                  {score}/{quizData.length}
                </div>
                <div className="text-2xl font-semibold text-gray-700 mb-2">
                  {percentage}% Correct
                </div>
                <div className={`text-lg font-medium ${messageColor}`}>
                  {message}
                </div>
              </div>

              <div className="flex gap-4 justify-center">
                <button
                  onClick={restartQuiz}
                  className="bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-900 hover:to-black text-white px-6 py-3 rounded-lg font-semibold transition-all transform hover:scale-105"
                >
                  Retake Training
                </button>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center gap-2">
              <Shield className="w-6 h-6 text-gray-800" />
              Key Phishing Prevention Guidelines
            </h3>
            <div className="space-y-4 text-gray-700">
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Verify sender domains:</strong> Always check that emails come from official @asu.edu addresses</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Be skeptical of urgency:</strong> Phishing emails often create false urgency to pressure quick action</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Hover before clicking:</strong> Check link destinations before clicking by hovering over them</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Never share credentials:</strong> ASU will never ask for your password via email</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Report suspicious emails:</strong> Forward phishing attempts to security@asu.edu immediately</p>
              </div>
              <div className="flex gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
                <p><strong>Watch for typos:</strong> Look for misspelled domains and poor grammar</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const currentEmail = quizData[currentQuestion];

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-50 to-red-50 p-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6 bg-white rounded-xl shadow-lg p-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <img 
              src="/asu-logo.png" 
              alt="Arizona State University" 
              className="h-10 object-contain"
            />
            <div>
              <h1 className="text-xl font-bold text-gray-900">Phishing Awareness Training</h1>
              <p className="text-sm text-gray-600">Information Security Education</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600">Question</div>
            <div className="text-2xl font-bold text-gray-800">
              {currentQuestion + 1}/{quizData.length}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
          <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6">
            <div className="flex items-center gap-3 text-white">
              <Mail className="w-6 h-6" />
              <h2 className="text-xl font-semibold">Email Scenario {currentQuestion + 1}</h2>
            </div>
          </div>

          <div className="p-8">
            <div className="bg-gray-50 rounded-lg p-6 mb-6 border border-gray-200 font-mono text-sm whitespace-pre-wrap">
              {currentEmail.email}
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Is this email legitimate or a phishing attempt?
              </h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => handleAnswer(true)}
                  disabled={answered}
                  className={`p-6 rounded-xl border-2 transition-all transform hover:scale-105 ${
                    answered
                      ? selectedAnswer === true
                        ? currentEmail.isPhishing
                          ? 'bg-green-50 border-green-500'
                          : 'bg-red-50 border-red-500'
                        : currentEmail.isPhishing && 'bg-green-50 border-green-500'
                      : 'border-gray-300 hover:border-red-500 hover:bg-red-50'
                  } ${answered && 'cursor-not-allowed'}`}
                >
                  <div className="flex items-center gap-3">
                    <XCircle className="w-8 h-8 text-red-600" />
                    <div className="text-left">
                      <div className="font-bold text-lg text-gray-900">Phishing</div>
                      <div className="text-sm text-gray-600">This is a malicious email</div>
                    </div>
                  </div>
                </button>

                <button
                  onClick={() => handleAnswer(false)}
                  disabled={answered}
                  className={`p-6 rounded-xl border-2 transition-all transform hover:scale-105 ${
                    answered
                      ? selectedAnswer === false
                        ? !currentEmail.isPhishing
                          ? 'bg-green-50 border-green-500'
                          : 'bg-red-50 border-red-500'
                        : !currentEmail.isPhishing && 'bg-green-50 border-green-500'
                      : 'border-gray-300 hover:border-green-500 hover:bg-green-50'
                  } ${answered && 'cursor-not-allowed'}`}
                >
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-8 h-8 text-green-600" />
                    <div className="text-left">
                      <div className="font-bold text-lg text-gray-900">Legitimate</div>
                      <div className="text-sm text-gray-600">This is a safe email</div>
                    </div>
                  </div>
                </button>
              </div>
            </div>

            {answered && (
              <div className="mt-6">
                <div className={`rounded-xl p-6 mb-4 ${
                  selectedAnswer === currentEmail.isPhishing
                    ? 'bg-green-50 border-2 border-green-500'
                    : 'bg-red-50 border-2 border-red-500'
                }`}>
                  <div className="flex items-center gap-2 mb-3">
                    {selectedAnswer === currentEmail.isPhishing ? (
                      <>
                        <CheckCircle className="w-6 h-6 text-green-600" />
                        <h4 className="font-bold text-lg text-green-800">Correct!</h4>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-6 h-6 text-red-600" />
                        <h4 className="font-bold text-lg text-red-800">Incorrect</h4>
                      </>
                    )}
                  </div>
                  
                  <p className="text-gray-700 mb-4">{currentEmail.explanation}</p>
                  
                  {currentEmail.isPhishing ? (
                    <div className="bg-white rounded-lg p-4 border border-red-200">
                      <h5 className="font-semibold text-red-800 mb-2 flex items-center gap-2">
                        <AlertCircle className="w-5 h-5" />
                        Red Flags:
                      </h5>
                      <ul className="space-y-1">
                        {currentEmail.redFlags.map((flag, idx) => (
                          <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                            <span className="text-red-500">✗</span> {flag}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <h5 className="font-semibold text-green-800 mb-2 flex items-center gap-2">
                        <CheckCircle className="w-5 h-5" />
                        Good Signs:
                      </h5>
                      <ul className="space-y-1">
                        {currentEmail.goodSigns.map((sign, idx) => (
                          <li key={idx} className="text-sm text-gray-700 flex items-center gap-2">
                            <span className="text-green-500">✓</span> {sign}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <button
                  onClick={nextQuestion}
                  className="w-full bg-gradient-to-r from-gray-800 to-gray-900 hover:from-gray-900 hover:to-black text-white py-4 rounded-xl font-semibold transition-all transform hover:scale-105 flex items-center justify-center gap-2"
                >
                  {currentQuestion < quizData.length - 1 ? (
                    <>
                      Next Question
                      <ArrowRight className="w-5 h-5" />
                    </>
                  ) : (
                    'View Results'
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="mt-6 bg-white rounded-xl shadow-lg p-4">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-600">Training Progress</div>
            <div className="text-sm font-semibold text-gray-800">
              Score: {score}/{currentQuestion + (answered ? 1 : 0)}
            </div>
          </div>
          <div className="mt-2 bg-gray-200 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-gray-800 to-gray-900 h-2 rounded-full transition-all duration-300"
              style={{ width: `${((currentQuestion + 1) / quizData.length) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PhishingQuiz;