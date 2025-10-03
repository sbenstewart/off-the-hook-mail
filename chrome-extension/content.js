// A variable to store the content of the last email we processed.
let lastEmailContent = null;
let debounceTimeout = null;

// Utility function to introduce a deliberate delay
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function analyzeAndDisplay(emailData) {
  try {
    // 1. Show the "Analyzing..." message instantly when analysis starts
    updateBanner({ is_phish: false, message: "Analyzing email for phishing indicators...", initial: true });

    // 2. Start the API request and the timer simultaneously
    const apiPromise = fetch('http://127.0.0.1:5000/analyze-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(emailData),
    }).then(response => response.json());

    const minWaitPromise = wait(2000); // Wait for a minimum of 2 seconds

    // 3. Wait for BOTH the API result AND the minimum wait time to complete
    const [result] = await Promise.all([apiPromise, minWaitPromise]);
    
    console.log("Server Response:", result);
    
    // Send a message to the background script to update stats
    chrome.runtime.sendMessage({
      action: 'updateStats',
      is_phish: result.is_phish
    });

    // 4. Update the banner with the final analysis result
    updateBanner(result);

  } catch (error) {
    console.error("Error communicating with server:", error);
    updateBanner({ is_phish: false, message: "Model connection failed. Check API token and server logs." });
  }
}

function updateBanner(result) {
  const banner = document.querySelector('.simple-banner');
  if (banner) {
    let bgColor, borderColor, textColor, message;

    // --- Google/Material Design Color Palette ---
    if (result.initial) {
        bgColor = '#fffde7'; 
        borderColor = '#fdd835'; 
        textColor = '#5f6368'; 
        message = `⏳ ${result.message}`;
    } else if (result.is_phish) {
      // Phishing Detected (Warning State - Red)
      bgColor = '#fef7f7'; 
      borderColor = '#f28b82'; 
      textColor = '#d93025'; 
      message = `🚨 HIGH RISK: ${result.message}`;
    } else {
      // Legitimate (Safe State - Blue)
      bgColor = '#e8f0fe'; 
      borderColor = '#8ab4f8'; 
      textColor = '#1a73e8'; 
      message = `✅ SAFE: ${result.message}`;
    }

    // Apply color and text updates
    banner.style.backgroundColor = bgColor;
    banner.style.borderColor = borderColor;
    banner.style.color = textColor;
    
    // Update the message content in the dedicated text span
    const messageSpan = banner.querySelector('.banner-message');
    if (messageSpan) {
        messageSpan.textContent = message;
    }
    
    // Update the color of the branding text
    const brandingText = banner.querySelector('.branding-text');
    if (brandingText) {
        brandingText.style.color = textColor;
    }
  }
}

function injectBanner() {
  const emailView = document.querySelector('div[role="main"]');
  if (emailView && emailView.querySelector('.a3s.aiL') && !emailView.querySelector('.simple-banner')) {
    const banner = document.createElement('div');
    
    // Apply Material Design structural styling
    banner.classList.add('simple-banner');
    banner.style.cssText = `
      padding: 12px 16px; 
      margin-bottom: 12px;
      margin-left: 10px;
      border-radius: 8px;
      border: 1px solid;
      box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.1);
      
      display: flex;
      flex-direction: column;
      font-size: 14px;
      line-height: 1.4;
      font-family: Roboto, Arial, sans-serif;
    `;
    
    // Inner HTML structure for the single-line message and footer
    banner.innerHTML = `
        <div class="banner-footer" style="display: flex; align-items: center; justify-content: space-between;">
            <!-- Left Side: Dynamic Message -->
            <span class="banner-message" style="font-weight: 500;">Analyzing email for phishing indicators...</span>
            
            <!-- Right Side: Powered By -->
            <div style="display: flex; align-items: center; white-space: nowrap; margin-left: 12px;">
                <span class="branding-text" style="font-weight: 400; font-size: 11px; margin-right: 6px; opacity: 0.8; color: currentColor;">Powered by Off-the-Hook</span>
                <img src="${chrome.runtime.getURL('images/icon-16.png')}" alt="Off-the-Hook Logo" style="width: 16px; height: 16px; display: block; opacity: 0.8;">
            </div>
        </div>
    `;

    emailView.prepend(banner);
    
    // Set initial analyzing state right after injection
    updateBanner({ initial: true, message: "Analyzing email for phishing indicators..." });
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
        
        injectBanner();
        
        analyzeAndDisplay(currentEmailContent);
      }
    }
  }, 300);
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
