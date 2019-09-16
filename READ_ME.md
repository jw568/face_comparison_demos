# Some fun face comparison demos I built!

## **Note: you will need to provide your own Configurations.py file with proper AWS keys to run AWS Rekognition.**

1. character_comparison_demo.py is a python program that compares the user’s face to celebrity faces from a movie or TV show, then morphs and swaps the user’s face with the face of the celebrity with the highest scoring similarity.
To run the character comparison demo, go to terminal and use the following command:
python character_comparison_demo.py [optional: --email=user's email] [THEME FOLDER]
examples: 
`python character_comparison_demo.py --email=jason.wang@gmail.com GoT`
`python character_comparison_demo.py Avengers`

Output will be saved to a folder called entertainment_results, with the results saved as [user's_email.jpg]

2. Split_four_new.py takes a picture of a user initially automatically (make sure there is only one face in the frame for this to work), then compares a live feed to that initial image, a linkedin image, and a random celebrity. To run the four screen split demo, just type:

`python split_four_new.py`

Can also change the LinkedIn photo by changing linkedin.jpg in the directory "split_four"

3. MTCNN_detection_demo.py draws the bounding box and key points of faces using MTCNN, just type:

`python MTCNN_detection_demo.py`

Press space bar to kill program

4. face_POI_demo.py takes a jpg image of a person and tries to detects that person in a live feed
- include a file called "target_face.jpg" in the directory, then run

`python face_POI_demo.py`
