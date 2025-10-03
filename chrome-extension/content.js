// A variable to store the content of the last email we processed.
let lastEmailContent = null;
let debounceTimeout = null;

// Function to send email data to the server and handle the response
async function analyzeAndDisplay(emailData) {
  try {
    const response = await fetch('http://127.0.0.1:5000/analyze-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(emailData),
    });

    const result = await response.json();
    console.log("Server Response:", result);
    
    // Send a message to the background script to update stats
    chrome.runtime.sendMessage({
      action: 'updateStats',
      is_phish: result.is_phish
    });

    // Update the banner based on the server's response
    updateBanner(result);

  } catch (error) {
    console.error("Error communicating with server:", error);
    updateBanner({ is_phish: false, message: "Could not connect to server." });
  }
}

// Function to update the banner with the analysis result
function updateBanner(result) {
  const existingBanner = document.querySelector('.simple-banner');
  if (existingBanner) {
    if (result.is_phish) {
      existingBanner.style.backgroundColor = '#fce8e8';
      existingBanner.style.border = '1px solid #c71c1c';
      existingBanner.style.color = '#c71c1c';
      existingBanner.textContent = `🚨 WARNING: ${result.message}`;
    } else {
      existingBanner.style.backgroundColor = '#e3f2fd';
      existingBanner.style.border = '1px solid #2196f3';
      existingBanner.style.color = '#1976d2';
      existingBanner.textContent = `✅ Result: ${result.message}`;
    }
  }
}

function processEmail() {
  clearTimeout(debounceTimeout);
  
  debounceTimeout = setTimeout(() => {
    const subjectElement = document.querySelector('h2.hP');
    const senderEmailElement = document.querySelector('.gD');
    const bodyElement = document.querySelector('.a3s.aiL');
    
    if (subjectElement && senderEmailElement && bodyElement) {
      const subject = subjectElement.textContent.trim();
      const senderEmail = senderEmailElement.getAttribute('email');
      const body = bodyElement.textContent.trim();
      
      const currentEmailContent = { subject, senderEmail, body };
      
      if (JSON.stringify(currentEmailContent) !== JSON.stringify(lastEmailContent)) {
        console.log("--- New Email Detected. Sending to server... ---");
        
        lastEmailContent = currentEmailContent;
        
        // Inject the initial banner
        injectBanner();
        
        // Send the data to the server for analysis
        analyzeAndDisplay(currentEmailContent);
      }
    }
  }, 300);
}

function injectBanner() {
  const emailView = document.querySelector('div[role="main"]');
  if (emailView && emailView.querySelector('.a3s.aiL') && !emailView.querySelector('.simple-banner')) {
    const banner = document.createElement('div');
    banner.style.cssText = `
      background-color: #e3f2fd;
      border: 1px solid #2196f3;
      color: #1976d2;
      padding: 10px;
      margin-bottom: 10px;
      font-weight: bold;
      border-radius: 5px;
      text-align: center;
    `;
    banner.textContent = "Analyzing email for phishing indicators...";
    banner.classList.add('simple-banner');
    emailView.prepend(banner);
  }
}

const observer = new MutationObserver((mutations) => {
  mutations.forEach(mutation => {
    if (mutation.addedNodes.length || mutation.removedNodes.length) {
      processEmail();
    }
  });
});

const gmailAppContainer = document.body;
if (gmailAppContainer) {
  observer.observe(gmailAppContainer, { childList: true, subtree: true });
}