function updatePopupStats() {
  chrome.storage.local.get(['totalScans', 'phishDetected'], (result) => {
    document.getElementById('total-scans-value').textContent = result.totalScans || 0;
    document.getElementById('phish-detected-value').textContent = result.phishDetected || 0;
  });
}

document.addEventListener('DOMContentLoaded', () => {
  updatePopupStats();
  
  document.getElementById('startGameButton').addEventListener('click', () => {
    // This command opens a new tab and loads your React app on localhost
    chrome.tabs.create({ url: 'http://localhost:5173' });
  });

  document.getElementById('startQuizButton').addEventListener('click', () => {
    // Navigates to the Quiz route
    chrome.tabs.create({ url: 'http://localhost:3000' });
  });

});