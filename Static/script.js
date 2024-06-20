document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const processedImage = document.getElementById("processed-image");
    const faceDetails = document.getElementById("face-details");
    const startCameraButton = document.getElementById("start-camera");
    const stopCameraButton = document.getElementById("stop-camera");
    const cameraStream = document.getElementById("camera-stream");
    const realTimeFaceDetails = document.getElementById("real-time-face-details");
    let videoFeed = document.createElement("video");
    let stream = null;
    let recognitionInterval = null;

    uploadForm.addEventListener("submit", (event) => {
        event.preventDefault();
        const formData = new FormData();
        const file = fileInput.files[0];
        
        if (!file) {
            alert("Please select a file!");
            return;
        }
        
        formData.append("file", file);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const imageUrl = "data:image/jpeg;base64," + data.image;
            processedImage.src = imageUrl;
            faceDetails.innerHTML = "";  // Clear previous face details
            if (data.faces && data.faces.length > 0) {
                data.faces.forEach(face => {
                    const faceInfo = `<p>Face: ${face[4]}</p>`;
                    faceDetails.innerHTML += faceInfo;
                });
            } else {
                faceDetails.innerHTML = "<p>No faces detected.</p>";
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("An error occurred while uploading the image: " + error.message);
        });
    });

    startCameraButton.addEventListener("click", async () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoFeed.srcObject = stream;
                videoFeed.play();
                cameraStream.srcObject = stream;
                recognitionInterval = setInterval(captureFrame, 1000);  
            } catch (error) {
                console.error('Error accessing the camera: ', error);
            }
        } else {
            alert('getUserMedia is not supported in this browser.');
        }
    });

    stopCameraButton.addEventListener("click", () => {
        if (stream) {
            let tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoFeed.srcObject = null;
            clearInterval(recognitionInterval);
        }
    });

    async function captureFrame() {
        if (videoFeed.readyState === videoFeed.HAVE_ENOUGH_DATA) {
            const canvas = document.createElement("canvas");
            canvas.width = videoFeed.videoWidth;
            canvas.height = videoFeed.videoHeight;
            const context = canvas.getContext("2d");
            context.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL("image/jpeg");

            try {
                const response = await fetch("/upload", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ image: imageData })
                });

                if (response.ok) {
                    const data = await response.json();
                    realTimeFaceDetails.innerHTML = "";  // Clear previous face details
                    if (data.faces && data.faces.length > 0) {
                        data.faces.forEach(face => {
                            const faceInfo = `<p>Face: ${face[4]}</p>`;
                            realTimeFaceDetails.innerHTML += faceInfo;
                        });
                    } else {
                        realTimeFaceDetails.innerHTML = "<p>No faces detected.</p>";
                    }
                } else {
                    console.error("Error in processing frame");
                }
            } catch (error) {
                console.error("Error:", error);
            }
        }
    }
});
