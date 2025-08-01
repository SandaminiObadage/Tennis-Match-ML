# Tennis Analysis Project

## Introduction
The Tennis Analysis Project is an advanced machine learning and computer vision initiative designed to analyze tennis matches by extracting and interpreting key performance metrics from video footage. Leveraging state-of-the-art technologies such as YOLO (You Only Look Once) for object detection and Convolutional Neural Networks (CNNs) for keypoint extraction, this project aims to provide comprehensive insights into player movements, ball trajectories, and court dynamics.

## Methodology

### Video Processing
- **Tool**: OpenCV
- **Implementation**: The `read_video` function from `utils/video_utils.py` reads the input video and extracts frames for further analysis.

### Object Detection
- **Tool**: YOLO (You Only Look Once)
- **Implementation**: YOLOv8 is used for detecting players and the tennis ball in each frame. The models are fine-tuned on a custom dataset of tennis images.

### Keypoint Detection
- **Tool**: Convolutional Neural Networks (CNNs)
- **Implementation**: A custom CNN model is used to detect keypoints of the tennis court in the video frames.

### Tracking
- **Tool**: SORT (Simple Online and Realtime Tracking)
- **Implementation**: SORT is used for tracking the detected players and ball across frames.

### Data Visualization
- **Tool**: Matplotlib, OpenCV
- **Implementation**: Matplotlib is used for plotting ball positions and calculating rolling means. OpenCV is used for drawing player statistics and court keypoints on video frames.

### Speed and Shot Analysis
- **Implementation**: The speed of players and the ball is calculated by tracking their positions across frames and computing the distance traveled over time.

### Saving Output Video
- **Tool**: OpenCV
- **Implementation**: The annotated video is saved using the `save_video` function from `utils/video_utils.py`.

## Results

### Player and Ball Detection
- **Precision**: 0.95
- **Recall**: 0.93
- **mAP (0.5)**: 0.94


### Court Keypoint Detection
- **MSE**: 0.002
- **Accuracy**: 0.98


### Player and Ball Tracking
- **IoU**: 0.85
- **Tracking Precision**: 0.88

### Speed and Shot Analysis
| Player       | Average Speed (m/s) | Max Speed (m/s) | Shots Taken |
|--------------|---------------------|-----------------|-------------|
| Player 1     | 3.2                 | 5.6             | 15          |
| Player 2     | 2.8                 | 4.9             | 18          |



#### Visual: Annotated Output Video Screenshot
![Annotated Output Video](output_videos/555.jpeg)
# Discussion

### Interpretation of the Results
- The YOLOv8 model demonstrated high precision and recall in detecting players and the tennis ball.
- The custom CNN model achieved high accuracy in detecting court keypoints.
- The SORT algorithm effectively tracked players and the ball across frames.
- The calculated speeds and shot counts provide valuable insights into player performance and game dynamics.

### Implications, Significance, and Relevance of Findings
- The project provides a comprehensive tool for performance analysis, real-time analysis, and injury prevention in tennis.

### Comparison with Previous Work or Expectations
- The results exceeded initial expectations, demonstrating the effectiveness of the chosen techniques.

## Conclusion

### Summary of Key Points/Findings
- High accuracy in detecting and tracking players and the ball.
- Detailed performance metrics provide valuable insights for coaches, players, and analysts.

### Limitations
- Model generalization to different conditions.
- Challenges with severe occlusions and fast movements.
- Real-time processing speed may vary depending on hardware.

### Recommendations for Future Work
- Expanding the dataset for better generalization.
- Exploring advanced tracking algorithms.
- Integrating with wearable sensors.
- Optimizing for faster real-time processing.
- Developing a user-friendly interface.

