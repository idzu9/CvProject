# CvProject
The final project after taking the Computer Vision course

# What it does 
It is a small utility that allows to use SAM model to process the video frame by frame using its text promt feature
1. User is able to pick a video file from the file system and open it 
2. Pause/Resume - user can manipulate the playback of the video using 'SPACEBAR' button
3. Frame forward/backward - user can make a framestep in both direction using 'ARROWS' buttons
4. User is able to enter the promt mode by pressing 'T', type the prompt and pressing 'ENTER' will submit the prompt to SAM model
5. As a result user will receive a segmentation results depicted on the frame
6. User can restore initial frame by pressing corresponding button on UI
7. User can isolate only segmented results on the frame by pressing corresponding button on UI

# How to use
1. Execute Main.py file to start working with the file
2. Closing file dialogue window will terminate the application
3. Closing opened video will trigger new file dialogue windows to be opened
4. 'Q' key press will close the app
5. If user switches to the next frame all the segmentation results will be cleared out and not carried to the next frame.

# Further improvements
1. Implementing UI with the help of other libraries instead of relying on OpenCV, since it is not flexible
2. Implementing SAM process in some kind of layer abstraction, allowing to scale the application with other models support and different kind of frame processing
3. Implementing ability for user to add coordinates to segment the objects,
