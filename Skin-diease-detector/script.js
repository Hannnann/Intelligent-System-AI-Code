// script.js

const API_URL = "http://<ngrok_public_url>/predict"; // Replace with actual ngrok URL

// Handle dragover event
function handleDragOver(event) {
    event.preventDefault();
    document.getElementById("dropArea").style.backgroundColor = "#e1f7e1"; // Highlight on drag
}

// Handle dragleave event
function handleDragLeave(event) {
    document.getElementById("dropArea").style.backgroundColor = "#f9f9f9"; // Reset background color
}

// Handle drop event (image drop)
function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
        previewImage(file);
    }
}

// Preview the image
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const imgElement = document.getElementById("previewImage");
        imgElement.src = e.target.result;
        document.getElementById("previewContainer").classList.remove("hidden");

        // Change the drop area text to "Image Selected"
        document.getElementById("dropText").textContent = "";
    }
    reader.readAsDataURL(file);
    document.getElementById("imageInput").files = event.dataTransfer.files; // Set the file input field with dropped file
}

// Trigger file input when the drag area is clicked
document.getElementById("dropArea").addEventListener("click", () => {
    document.getElementById("imageInput").click();
});

// Handle the file input change (when a file is selected via the input)
document.getElementById("imageInput").addEventListener("change", () => {
    const file = document.getElementById("imageInput").files[0];
    if (file) {
        previewImage(file);
    }
});

// Function to handle image upload and send to the API
function uploadImage() {
    const imageInput = document.getElementById("imageInput");
    const file = imageInput.files[0];

    if (!file) {
        alert("Please select an image!");
        return;
    }

    // Show loading indicator
    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");
    document.getElementById("error").classList.add("hidden");

    // Prepare the form data to send to the API
    const formData = new FormData();
    formData.append("image", file);

    // Make the API call to FastAPI server
    fetch(API_URL, {
        method: "POST",
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            window.alert(JSON.stringify(data));
            // Hide loading indicator
            document.getElementById("loading").classList.add("hidden");

            if (data.predictions) {
                // Show the prediction results
                const diseaseType = data.predictions[0].label;
                const confidence = (data.predictions[0].confidence * 100).toFixed(2);
                const disclaimer = data.disclaimer;

                document.getElementById("diseaseType").textContent = `Disease: ${diseaseType}`;
                document.getElementById("confidence").textContent = `Confidence: ${confidence}%`;
                document.getElementById("disclaimer").textContent = disclaimer;

                document.getElementById("result").classList.remove("hidden");
            } else {
                // Handle error if predictions are not received
                document.getElementById("error").classList.remove("hidden");
            }
        })
        .catch(error => {
            // Hide loading indicator and show error message
            document.getElementById("loading").classList.add("hidden");
            document.getElementById("error").classList.remove("hidden");
        });
}

// Attach the drag and drop event listeners
const dropArea = document.getElementById("dropArea");
dropArea.addEventListener("dragover", handleDragOver);
dropArea.addEventListener("dragleave", handleDragLeave);
dropArea.addEventListener("drop", handleDrop);
