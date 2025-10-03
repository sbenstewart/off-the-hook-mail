# off-the-hook-mail: 
## Chrome Extension & ML Analysis System
This project combines a Chrome extension, a Python backend, and a React-based educational game to create a comprehensive system for real-time email analysis and cybersecurity education.

## Project Components
The system is split into three main, independent components that communicate with each other:

* Chrome Extension (chrome-extension/): The client-side component that injects a banner into Gmail and serves as the UI for tracking statistics and launching the game.

* Python Flask Server (python-server/): The backend service that receives email content, simulates an ML analysis delay, and returns a classification result.

* React Game (game/): A separate educational application that allows users to practice identifying phishing attempts.

## Repository Structure
The project is organized into three top-level folders:

```
/phishing-detector-project
|-- /chrome-extension       <-- LOAD THIS FOLDER IN CHROME
|   |-- manifest.json       (Configuration)
|   |-- background.js       (Storage Bridge)
|   |-- content.js          (Gmail UI Interaction)
|   |-- popup.html / popup.js (Stats UI)
|   |-- /images
|
|-- /python-server
|   |-- server.py           (Flask Backend)
|
|-- /game                   <-- REACT GAME DIRECTORY
|   |-- package.json
|   |-- /src
|   |-- ...
```

## Setup and Installation
### Prerequisites
* Node.js and npm: For running the React game.
* Python 3: For running the Flask server.

* Flask: Install the required Python library(CORS is essential for communication with the extension).: 
pip install Flask flask-cors 

### Step 1: Start the React Game
The game is crucial for the extension's "Play Game" button to function correctly.
* Open a terminal and navigate to the game/ directory.
* Install dependencies and start the development server:
```
cd game
npm install
npm start
```
The game should now be running at http://localhost:3000. Keep this terminal running.

### Step 2: Start the Python Flask Server
The server handles all the analysis requests from the extension.

* Open a second terminal and navigate to the python-server/ directory.
* Start the Flask application:
```
cd python-server
python server.py
```
* The server should be running at http://127.0.0.1:5000. Keep this terminal running.

(Note: This endpoint currently contains mock logic. You would replace the time.sleep(5) delay and the simple keyword check with your actual ML model logic.)

### Step 3: Load the Chrome Extension
* Open Google Chrome and navigate to chrome://extensions.
* Enable Developer mode (toggle in the top right).
* Click the Load unpacked button.
* Select the /chrome-extension folder.
* The extension is now installed.

## Usage
* Open Gmail: Navigate to your Gmail inbox.
* Analysis: Click on any email.
* The content.js script will detect the new email.
* A loading banner will appear, and the extension will make an asynchronous POST request to http://127.0.0.1:5000/analyze-email.
* After the 5-second simulated delay, the banner will update with the mock analysis result.
* **View Stats:** Click the extension's icon in the toolbar. The popup will display the Total Emails Scanned and Phishing Emails Detected (which are saved in Chrome's local storage).
* **Play Game:** Click the Play Game button in the popup to launch the React game in a new, full-screen browser tab.