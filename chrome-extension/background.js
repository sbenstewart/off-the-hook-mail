// This listener runs the first time the extension is installed.
// It initializes storage to prevent "undefined" errors.
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    totalScans: 0,
    phishDetected: 0
  });
});

// This script acts as a bridge, listening for messages from the content script.
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // We use an asynchronous function to handle the promise from the storage API.
  (async () => {
    if (request.action === 'updateStats') {
      try {
        // Use async/await to ensure the storage data is fetched before we use it.
        const result = await chrome.storage.local.get(['totalScans', 'phishDetected']);
        
        let totalScans = (result.totalScans || 0) + 1;
        let phishDetected = (result.phishDetected || 0);

        if (request.is_phish) {
          phishDetected += 1;
        }
        
        // Save the updated stats to storage.
        await chrome.storage.local.set({ totalScans, phishDetected });
        sendResponse({ status: 'success' });
      } catch (error) {
        console.error("Error updating stats:", error);
        sendResponse({ status: 'error', message: error.toString() });
      }
    }
  })();
  
  // Return true to indicate that we will send a response asynchronously.
  return true; 
});