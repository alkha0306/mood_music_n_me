# Facial Emotion based Playlist Recommender App
The project contains a flask based app that detects the current emotional state of the user by using the image captured by the user and detecting the emotion based on the users facial expressions.
We use a pre trained model for face detection and FER library for facial emotion recognition.
The OpenCV library is used to help enable capturing and processing the users picture.

## Working
Here's how it works.

On opening the app it helps you navigate to a capture page where the user can take his/her picture for further analysis.

![Screenshot-Output](frontend-img/frontend_1.png?raw=true "Title")

![Screenshot-Output](frontend-img/frontend_2.png?raw=true "Title")

When user clicks on capture the users picture is taken and a message is displayed telling the user that they have successfully taken the picture and they can now click on another button to move to emotion detection.


![Screenshot-Output](frontend-img/frontend_3.png?raw=true "Title")

The app analyses the picture using the trained model and the screen displays the detected emotion as result.

![Screenshot-Output](frontend-img/frontend_4.png?raw=true "Title")

User is now redirected to a playlist based on their emotions.

![Screenshot-Output](frontend-img/frontend_6.png?raw=true "Title")
