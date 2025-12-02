const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const matchNameEl = document.getElementById('match-name');
const scoreBarEl = document.getElementById('score-bar');
const scoreTextEl = document.getElementById('score-text');
const matchImageEl = document.getElementById('match-image');
const placeholderTextEl = document.getElementById('placeholder-text');

let isProcessing = false;
let lastMatchName = "...";
let currentScore = 0;

// Access Webcam
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please allow camera permissions.");
    }
}

// Send frame to server
async function sendFrame() {
    if (isProcessing) return;
    isProcessing = true;

    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to base64
    const dataURL = canvas.toDataURL('image/jpeg', 0.7); // 0.7 quality to save bandwidth

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: dataURL })
        });

        const data = await response.json();

        if (data.match_name) {
            updateUI(data.match_name, data.score);
        }
    } catch (err) {
        console.error("Error sending frame:", err);
    } finally {
        isProcessing = false;
        // Schedule next frame. 
        // Adjust delay to balance performance/lag. 100ms = ~10fps max
        setTimeout(sendFrame, 100);
    }
}

function updateUI(name, score) {
    // Smooth score update could be done here, but for now direct update

    // Threshold to show match
    const THRESHOLD = 0.4; // Slightly lower than backend to be responsive

    if (score > THRESHOLD) {
        matchNameEl.textContent = name;
        scoreTextEl.textContent = Math.round(score * 100) + "%";
        scoreBarEl.style.width = (score * 100) + "%";

        // Color coding
        if (score > 0.8) scoreBarEl.style.backgroundColor = "#00ff00";
        else if (score > 0.6) scoreBarEl.style.backgroundColor = "#ffff00";
        else scoreBarEl.style.backgroundColor = "#ff0000";

        // Update Image
        if (name !== lastMatchName || matchImageEl.style.display === 'none') {
            matchImageEl.src = `/memes/${name}`;
            matchImageEl.style.display = 'block';
            placeholderTextEl.style.display = 'none';
            lastMatchName = name;
        }
    } else {
        scoreBarEl.style.width = (score * 100) + "%";
        scoreTextEl.textContent = Math.round(score * 100) + "%";
        // Don't hide image immediately to avoid flickering, just update score
    }
}

video.addEventListener('loadeddata', () => {
    sendFrame();
});

startCamera();
