// Function to update system status and apply the wave effect
function updateStatus() {
    fetch('/get_system_status')
        .then((response) => response.json())
        .then((data) => {
            const statusElement = document.getElementById('status');
            const circleElement = document.getElementById('listeningCircle');

            // Update the status text
            statusElement.innerText = data.status;

            // Add the wave effect when listening, remove it otherwise
            if (data.status === 'Listening') {
                circleElement.classList.add('pulse-wave');
            } else {
                circleElement.classList.remove('pulse-wave');
            }
        });
}

// Update the status every second
setInterval(updateStatus, 1000);



// Continuously update the user input and bot reply every second
setInterval(() => {
    fetch("/get_data")
        .then((response) => response.json())
        .then((data) => {
            document.getElementById("user_input").innerText = "You said: " + data.user_input;
            document.getElementById("bot_reply").innerText = "Bili: " + data.bot_reply;
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}, 1000);

// Continuously update the status every second
setInterval(updateStatus, 1000);

function startListening() {
    // Set the status to "Listening..."
    fetch("/set_listening_status/true", {
            method: "POST"
        })
        .then((response) => response.json())
        .then(() => {
            // Update the conversation and status dynamically
            document.getElementById("status").innerText = "Status: Listening...";
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

function stopListening() {
    // Set the status to "Ready"
    fetch("/set_listening_status/false", {
            method: "POST"
        })
        .then((response) => response.json())
        .then(() => {
            // Update the conversation and status dynamically
            document.getElementById("status").innerText = "Status: Listening...";
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}

// Continuously update the user input and bot reply every second
setInterval(() => {
    fetch("/get_data")
        .then((response) => response.json())
        .then((data) => {
            document.getElementById("user_input").innerText = "You said: " + data.user_input;
            document.getElementById("bot_reply").innerText = "Bili: " + data.bot_reply;
        })
        .catch((error) => {
            console.error("Error:", error);
        });
}, 1000);