import React, { useState, useEffect, useRef } from 'react';
import Papa from 'papaparse';

const InboxDefender = () => {
  const [gameState, setGameState] = useState('menu');
  const [score, setScore] = useState(0);
  const [lives, setLives] = useState(3);
  const [emails, setEmails] = useState([]);
  const [bullets, setBullets] = useState([]);
  const [playerPos, setPlayerPos] = useState(50);
  const [stats, setStats] = useState({
    legitimate_caught: 0,
    malicious_shot: 0,
    malicious_missed: 0,
    legitimate_missed: 0
  });
  const emailDataRef = useRef([]);
  const emailIdCounter = useRef(0);
  const lastSpawnRef = useRef(0);

  useEffect(() => {
    const loadEmails = async () => {
      try {
        const response = await fetch('/se_phishing_test_set.csv');
        const fileContent = await response.text();
        
        const parsed = Papa.parse(fileContent, {
          header: true,
          skipEmptyLines: true
        });
        
        emailDataRef.current = parsed.data.map(row => {
          const text = row.email_text || '';
          const fromMatch = text.match(/From: (.+?)[\n<]/);
          const lines = text.split('\n');
          const bodyLines = [];
          let foundSubject = false;
          
          for (const line of lines) {
            if (foundSubject && line.trim() && !line.startsWith('http') && !line.startsWith('From:') && !line.startsWith('To:')) {
              bodyLines.push(line.trim());
              if (bodyLines.length >= 2) break;
            }
            if (line.startsWith('Subject:')) {
              foundSubject = true;
            }
          }
          
          const body = bodyLines.join(' ');
          
          return {
            from: fromMatch ? fromMatch[1].trim() : 'Unknown Sender',
            body: body || 'Click here to verify your account',
            isMalicious: row.label === 'Malicious'
          };
        });
        
        console.log('Loaded emails:', emailDataRef.current.length);
      } catch (err) {
        console.error('Error loading emails:', err);
      }
    };
    loadEmails();
  }, []);

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (gameState !== 'playing') return;
      
      if (e.key === 'ArrowLeft') {
        setPlayerPos(prev => Math.max(5, prev - 5));
      } else if (e.key === 'ArrowRight') {
        setPlayerPos(prev => Math.min(95, prev + 5));
      } else if (e.key === ' ') {
        e.preventDefault();
        setBullets(prev => [...prev, { id: Date.now(), x: playerPos, y: 85 }]);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [gameState, playerPos]);

  useEffect(() => {
    if (gameState !== 'playing') return;

    const interval = setInterval(() => {
      const now = Date.now();
      
      if (now - lastSpawnRef.current > 2000 && emailDataRef.current.length > 0) {
        const randomEmail = emailDataRef.current[Math.floor(Math.random() * emailDataRef.current.length)];
        setEmails(prev => [...prev, {
          ...randomEmail,
          id: emailIdCounter.current++,
          x: Math.random() * 80 + 10,
          y: 0,
          speed: 0.5 + Math.random() * 0.5
        }]);
        lastSpawnRef.current = now;
      }

      setEmails(prev => prev.map(email => ({ ...email, y: email.y + email.speed })).filter(email => {
        if (email.y > 95) {
          if (email.isMalicious) {
            setStats(s => ({ ...s, malicious_missed: s.malicious_missed + 1 }));
          } else {
            setStats(s => ({ ...s, legitimate_missed: s.legitimate_missed + 1 }));
            setLives(l => l - 1);
          }
          return false;
        }
        return true;
      }));

      setBullets(prev => prev.map(bullet => ({ ...bullet, y: bullet.y - 2 })).filter(bullet => bullet.y > 0));

      setBullets(prevBullets => {
        const remainingBullets = [...prevBullets];
        
        setEmails(prevEmails => {
          const remainingEmails = [];
          
          for (const email of prevEmails) {
            let hit = false;
            
            for (let i = remainingBullets.length - 1; i >= 0; i--) {
              const bullet = remainingBullets[i];
              const distance = Math.sqrt(Math.pow(email.x - bullet.x, 2) + Math.pow(email.y - bullet.y, 2));
              
              if (distance < 5) {
                hit = true;
                remainingBullets.splice(i, 1);
                
                if (email.isMalicious) {
                  setScore(s => s + 100);
                  setStats(s => ({ ...s, malicious_shot: s.malicious_shot + 1 }));
                } else {
                  setScore(s => Math.max(0, s - 50));
                  setLives(l => l - 1);
                }
                break;
              }
            }
            
            if (!hit && !email.isMalicious && email.y > 85 && email.y < 95) {
              const inboxDistance = Math.abs(email.x - playerPos);
              if (inboxDistance < 8) {
                hit = true;
                setScore(s => s + 50);
                setStats(s => ({ ...s, legitimate_caught: s.legitimate_caught + 1 }));
              }
            }
            
            if (!hit) {
              remainingEmails.push(email);
            }
          }
          
          return remainingEmails;
        });
        
        return remainingBullets;
      });

    }, 50);

    return () => clearInterval(interval);
  }, [gameState, playerPos]);

  useEffect(() => {
    if (lives <= 0 && gameState === 'playing') {
      setGameState('gameover');
    }
  }, [lives, gameState]);

  const startGame = () => {
    setGameState('playing');
    setScore(0);
    setLives(3);
    setEmails([]);
    setBullets([]);
    setPlayerPos(50);
    setStats({ legitimate_caught: 0, malicious_shot: 0, malicious_missed: 0, legitimate_missed: 0 });
    lastSpawnRef.current = Date.now();
  };

  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1e1b4b 0%, #581c87 50%, #1e1b4b 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px',
      fontFamily: 'Arial, sans-serif'
    },
    button: {
      padding: '15px 40px',
      background: 'linear-gradient(90deg, #06b6d4 0%, #a855f7 100%)',
      color: 'white',
      border: 'none',
      borderRadius: '30px',
      fontSize: '20px',
      fontWeight: 'bold',
      cursor: 'pointer',
      boxShadow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    gameArea: {
      position: 'relative',
      background: 'rgba(15, 23, 42, 0.8)',
      borderRadius: '20px',
      border: '4px solid rgba(6, 182, 212, 0.3)',
      height: '600px',
      width: '100%',
      overflow: 'hidden'
    }
  };

  return (
    <div style={styles.container}>
      <div style={{width: '100%'}}>
        {gameState === 'menu' && (
          <div style={{textAlign: 'center', color: 'white'}}>
            <h1 style={{fontSize: '60px', background: 'linear-gradient(90deg, #22d3ee 0%, #a78bfa 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '10px'}}>
              Inbox Defender
            </h1>
            <p style={{fontSize: '20px', color: '#d1d5db', marginBottom: '40px'}}>Protect your inbox from phishing attacks!</p>
            
            <div style={{background: 'rgba(30, 41, 59, 0.5)', borderRadius: '20px', padding: '40px', border: '1px solid rgba(6, 182, 212, 0.2)', marginBottom: '40px'}}>
              <h2 style={{fontSize: '24px', color: '#22d3ee', marginBottom: '30px'}}>How to Play</h2>
              <div style={{textAlign: 'left', color: '#d1d5db', maxWidth: '600px', margin: '0 auto'}}>
                <p style={{marginBottom: '20px'}}><strong style={{color: 'white'}}>Shoot malicious emails</strong> - Use SPACE to fire at phishing attempts</p>
                <p style={{marginBottom: '20px'}}><strong style={{color: 'white'}}>Catch legitimate emails</strong> - Move with ARROW KEYS to collect safe emails at the bottom</p>
                <p><strong style={{color: 'white'}}>Protect your lives</strong> - Don't shoot good emails or miss bad ones!</p>
              </div>
            </div>

            <button onClick={startGame} style={styles.button}>Start Game</button>
          </div>
        )}

        {gameState === 'playing' && (
          <div>
            <div style={{display: 'flex', justifyContent: 'space-between', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '15px', padding: '20px', marginBottom: '20px', color: 'white'}}>
              <div>
                <span style={{fontSize: '28px', fontWeight: 'bold'}}>Score: {score}</span>
                <span style={{marginLeft: '20px'}}>Lives: {'❤️'.repeat(lives)}</span>
              </div>
              <div style={{fontSize: '14px', color: '#9ca3af'}}>
                <div>✓ Legit Caught: {stats.legitimate_caught}</div>
                <div>⚡ Phishing Shot: {stats.malicious_shot}</div>
              </div>
            </div>

            <div style={styles.gameArea}>
              {emails.map(email => (
                <div
                  key={email.id}
                  style={{
                    position: 'absolute',
                    left: `${email.x}%`,
                    top: `${email.y}%`,
                    width: '220px',
                    transform: 'translateX(-50%)',
                    background: email.isMalicious ? 'linear-gradient(135deg, #ef4444 0%, #f97316 100%)' : 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                    borderRadius: '10px',
                    padding: '12px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                    border: `2px solid ${email.isMalicious ? '#fca5a5' : '#6ee7b7'}`
                  }}
                >
                  <div style={{fontSize: '11px', fontWeight: 'bold', color: 'white', marginBottom: '5px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                    {email.isMalicious ? '⚠️ ' : '✓ '}From: {email.from}
                  </div>
                  <div style={{fontSize: '10px', color: 'rgba(255, 255, 255, 0.9)', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical'}}>
                    {email.body.substring(0, 60)}...
                  </div>
                </div>
              ))}

              {bullets.map(bullet => (
                <div
                  key={bullet.id}
                  style={{
                    position: 'absolute',
                    left: `${bullet.x}%`,
                    top: `${bullet.y}%`,
                    width: '8px',
                    height: '16px',
                    background: '#22d3ee',
                    borderRadius: '10px',
                    boxShadow: '0 0 10px rgba(34, 211, 238, 0.7)',
                    transform: 'translateX(-50%)'
                  }}
                />
              ))}

              <div style={{position: 'absolute', left: `${playerPos}%`, bottom: '20px', transform: 'translateX(-50%)'}}>
                <div style={{fontSize: '64px', filter: 'drop-shadow(0 0 10px rgba(34, 211, 238, 0.7))'}}>🛡️</div>
              </div>
            </div>

            <div style={{textAlign: 'center', marginTop: '15px', color: '#9ca3af', fontSize: '14px'}}>
              Use ← → to move | SPACE to shoot
            </div>
          </div>
        )}

        {gameState === 'gameover' && (
          <div style={{textAlign: 'center', color: 'white'}}>
            <h2 style={{fontSize: '50px', color: '#ef4444', marginBottom: '30px'}}>Game Over!</h2>
            
            <div style={{background: 'rgba(30, 41, 59, 0.5)', borderRadius: '20px', padding: '40px', border: '1px solid rgba(6, 182, 212, 0.2)', marginBottom: '30px'}}>
              <div style={{fontSize: '40px', fontWeight: 'bold', marginBottom: '30px'}}>Final Score: {score}</div>

              <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', maxWidth: '600px', margin: '0 auto'}}>
                <div style={{background: 'rgba(16, 185, 129, 0.2)', borderRadius: '10px', padding: '20px', border: '1px solid rgba(16, 185, 129, 0.3)'}}>
                  <div style={{color: '#10b981', fontWeight: 'bold', marginBottom: '10px'}}>Legitimate Emails</div>
                  <div>✓ Caught: {stats.legitimate_caught}</div>
                  <div style={{color: '#fca5a5'}}>✗ Missed: {stats.legitimate_missed}</div>
                </div>
                <div style={{background: 'rgba(239, 68, 68, 0.2)', borderRadius: '10px', padding: '20px', border: '1px solid rgba(239, 68, 68, 0.3)'}}>
                  <div style={{color: '#ef4444', fontWeight: 'bold', marginBottom: '10px'}}>Malicious Emails</div>
                  <div>⚡ Shot: {stats.malicious_shot}</div>
                  <div style={{color: '#fde047'}}>⚠ Missed: {stats.malicious_missed}</div>
                </div>
              </div>
            </div>

            <div style={{display: 'flex', gap: '20px', justifyContent: 'center'}}>
              <button onClick={startGame} style={styles.button}>Play Again</button>
              <button onClick={() => setGameState('menu')} style={{...styles.button, background: '#475569'}}>Main Menu</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InboxDefender;