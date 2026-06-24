AI-based Obstacle Detection and UAV Flight Control

This repository contains the source code developed for the Bachelor Thesis:

AI-based technique for obstacle detection and UAV flight control

The project implements a reactive obstacle avoidance prototype for a low-cost UAV using monocular RGB vision, remote real-time processing, YOLO object detection and SDK/API-based actuation.

The system follows the architecture:

Perception → Decision → Actuation

The UAV captures video using its onboard RGB camera and sends the stream to a remote computer. The computer performs object detection, estimates collision risk from image-space variables and sends corrective low-speed commands back to the drone through the SDK/API.

---

1. Project scope

This project is an experimental prototype developed for controlled validation. It is not a certified autonomous navigation system and must not be used in uncontrolled environments.

Main characteristics:

- UAV platform: DJI Tello / Tello-like UAV
- Camera: onboard monocular RGB camera
- Processing: remote/off-board computer
- Communication: WiFi / UDP
- Programming language: Python
- Main libraries: OpenCV, DJITelloPy, Ultralytics YOLO
- Detection model: YOLO11n
- Decision logic: bounding-box area, central safety region, temporal growth, detection persistence and obstacle-clear logic
- Actuation: conservative low-speed SDK/API commands

---

2. Repository contents

Recommended repository structure:

ai-obstacle-detection-uav-flight-control/
│
├── README.md
├── requirements.txt
│
├── src/
│   └── tfg_v6_5.py
│
└── docs/
    └── tfg_v6_5_line_by_line_guide.pdf

Main files:

- "src/tfg_v6_5.py": final implementation used for the experimental prototype.
- "docs/tfg_v6_5_line_by_line_guide.pdf": supplementary line-by-line explanation of the code.
- "requirements.txt": Python dependencies required to run the script.

---

3. Safety warning

This code controls a real UAV. It must only be executed under controlled conditions.

Before every test:

- Check that the drone battery is sufficiently charged.
- Check propellers and propeller guards.
- Check that the landing gear or support structure is correctly attached, if used.
- Verify that the test area is clear of people, fragile objects and obstacles not intended for the experiment.
- Keep manual supervision during the complete test.
- Keep enough free space around the UAV.
- Avoid outdoor tests under wind or unstable lighting conditions.
- Do not use aggressive speeds or untested modifications.
- Be ready to press the landing key or stop the experiment immediately.

The implemented system is a reactive obstacle avoidance prototype. It does not provide certified safety, full autonomy, SLAM, GPS navigation, metric 3D reconstruction or guaranteed obstacle clearance.

---

4. Software environment

The code was developed in a Windows environment using Anaconda and Spyder.

The original experimental environment used by the author was named:

ide_spyder

The exact installed packages may vary depending on the local setup. The essential dependencies are:

python
opencv-python
djitellopy
ultralytics
numpy

A typical installation can be performed with:

pip install opencv-python djitellopy ultralytics numpy

If a "requirements.txt" file is provided, install the dependencies using:

pip install -r requirements.txt

Important: the YOLO model file should be available before connecting to the Tello WiFi network, because the Tello WiFi usually does not provide internet access.

The final script uses:

MODEL_PATH = "yolo11n.pt"

Therefore, make sure that "yolo11n.pt" is available in the working directory or that Ultralytics can download it before connecting to the drone network.

---

5. Preparing the project folder

Create a local project folder, for example:

C:\Users\YourUser\UAV_Project\

Place the final script inside the folder:

C:\Users\YourUser\UAV_Project\tfg_v6_5.py

Or, if the repository structure is used:

C:\Users\YourUser\UAV_Project\src\tfg_v6_5.py

The script will automatically create an output directory for each experimental run. In the final version, the base output directory is:

tello_quantitative_trials_v6_5_bidirectional_vertical_escape

Inside that folder, each execution creates a timestamped run folder containing logs, frames and videos when enabled.

---

6. Opening Anaconda Prompt and activating the environment

Open Anaconda Prompt.

Activate the environment used for the project:

conda activate ide_spyder

Move to the folder where the script is stored:

cd C:\Users\YourUser\UAV_Project

If the script is inside "src", use:

cd C:\Users\YourUser\UAV_Project\src

Check that Python can import the main libraries:

python -c "import cv2; import djitellopy; import ultralytics; print('Environment OK')"

If this command fails, install the missing dependency before continuing.

---

7. WiFi connection with the drone

After the software environment is ready, connect the computer to the DJI Tello WiFi network.

Typical network name:

TELLO-XXXXXX

Use the Windows WiFi menu to connect to the drone network.

Important notes:

- The computer may lose internet access while connected to the drone WiFi.
- Download dependencies and the YOLO model before connecting to the drone.
- A USB WiFi antenna may improve the connection stability and range.
- Keep the computer close to the drone during the experiment.
- Avoid crowded WiFi environments when possible.

---

8. UDP ports

The script uses DJITelloPy, which internally communicates with the Tello through the standard Tello SDK UDP channels.

Typical Tello SDK ports:

8889   command channel
8890   state / telemetry channel
11111  video stream channel

Before running the script, make sure that no other Python script, Tello application or previous process is using these ports.

On Windows, the following commands can be used in Command Prompt to check whether a port is being used:

netstat -ano | findstr :8889
netstat -ano | findstr :8890
netstat -ano | findstr :11111

If a previous Python process is still running, close it before starting a new experiment.

It is also recommended to allow Python through the Windows firewall when prompted, otherwise video streaming or telemetry may fail.

---

9. Running the script

From Anaconda Prompt, with the correct environment activated and the computer connected to the Tello WiFi, run:

python tfg_v6_5.py

If the script is inside "src", run:

python src\tfg_v6_5.py

If Spyder is preferred, open Spyder from the same Anaconda environment and run the file from the editor.

When the script starts correctly:

1. The drone connects through DJITelloPy.
2. The battery level is printed.
3. The video stream starts.
4. A window opens using OpenCV.
5. The safety corridor and detection overlays are displayed on the video frame.

The script checks the battery level before testing. If the battery is below the configured minimum threshold, the script stops for safety.

---

10. Operator controls

The OpenCV window must be active for the keyboard controls to work.

Main keys:

T      take off
L      land
ENTER  start a new avoidance attempt
X      manually abort the current attempt
Q/ESC  quit the program

Operator labels after each attempt:

Y      mark previous attempt as successful
F      mark previous attempt as failed
U      mark previous attempt as unclear or invalid

Scenario selection keys:

1      centered person
2      left person
3      right person
4      centered car
5      centered person upward escape
6      centered person downward escape

Manual control keys, only when the drone is flying and no automatic attempt is active:

Arrow Up       move forward
Arrow Down     move backward
Arrow Left     move left
Arrow Right    move right
W              move up
S              move down
A              yaw left
D              yaw right

The operator must label each attempt before starting the next one. This is required to generate the "operator_trial_labels.csv" file.

---

11. Experimental workflow

Recommended execution sequence:

1. Place the drone in a safe and controlled test area.
2. Check battery, propellers, propeller guards and landing area.
3. Place the obstacle in the intended test position.
4. Open Anaconda Prompt.
5. Activate the environment:

conda activate ide_spyder

6. Move to the project folder.
7. Connect the computer to the Tello WiFi network.
8. Run the script:

python tfg_v6_5.py

9. Wait until the OpenCV window shows the video stream.
10. Press "T" to take off.
11. Select the scenario if needed.
12. Press "ENTER" to start an avoidance attempt.
13. Supervise the full manoeuvre.
14. Press "L" to land if required.
15. Label the attempt with "Y", "F" or "U".
16. Repeat the test if needed.
17. Press "Q" or "ESC" to close the program safely.

---

12. Output files

Each run generates a folder containing experimental logs and visual evidence.

Main output files:

run_config.json
events.csv
attempts.csv
detections.csv
telemetry.csv
operator_trial_labels.csv

If frame or video saving is enabled in the script, the following outputs may also be generated:

frames_raw/
frames_annotated/
video_raw.mp4
video_annotated.mp4

Purpose of each log:

- "run_config.json": stores the configuration used during the run.
- "events.csv": stores state transitions and relevant events.
- "attempts.csv": stores attempt-level results and automatic success information.
- "detections.csv": stores YOLO detections, confidence scores and bounding-box values.
- "telemetry.csv": stores battery, height, attitude and state feedback.
- "operator_trial_labels.csv": stores manual labels assigned by the operator after each attempt.

These files are used to analyse the Perception → Decision → Actuation behaviour after the flight.

---

13. Troubleshooting

The drone does not connect

Check that:

- The computer is connected to the Tello WiFi.
- No other application is connected to the drone.
- The drone battery is charged.
- The drone was powered on recently.
- The firewall is not blocking Python.

Restarting the drone and reconnecting the WiFi may solve the issue.

The video window does not open

Check that:

- OpenCV is installed.
- The video stream started correctly.
- The computer is still connected to the Tello WiFi.
- No previous Python process is still using the video port.

YOLO model cannot be loaded

Check that:

- "ultralytics" is installed.
- "yolo11n.pt" is available locally.
- The model was downloaded before connecting to the Tello WiFi.

The keyboard does not respond

Click on the OpenCV window before pressing the keys. The keyboard commands work only when the OpenCV window is active.

The drone behaves unstably

Stop the test immediately. Possible causes include:

- low battery,
- poor WiFi connection,
- indoor airflow,
- bad lighting,
- motion blur,
- excessive command delay,
- obstacle too close,
- reflective or textureless background.

Use low-speed tests and keep manual supervision.

---

14. Reproducibility notes

To reproduce the experiments, record at least:

- date and time of the run,
- script version,
- environment name,
- YOLO model used,
- drone battery level,
- test scenario,
- obstacle type,
- lighting conditions,
- distance to obstacle,
- operator labels,
- generated CSV logs,
- annotated frames or video evidence.

The final thesis analysis was based on controlled real-flight logs, annotated frames, telemetry, event records and operator-reviewed attempt labels.

---

15. Limitations

The system has several important limitations:

- The camera is monocular and does not provide direct metric depth.
- The distance estimate is a calibration-based proxy derived from bounding-box area.
- YOLO detections may fail under poor lighting, motion blur or occlusions.
- WiFi/UDP communication can introduce latency and packet loss.
- The Tello platform is lightweight and sensitive to wind, drift and battery level.
- The upward avoidance branch is an image-space heuristic and does not certify overhead clearance.
- The system is reactive and map-free. It does not perform SLAM, GPS navigation or global path planning.

The prototype must therefore be interpreted as a controlled academic implementation of reactive obstacle avoidance, not as a deployable autonomous navigation product.

---

16. Citation

This repository supports the Bachelor Thesis:

AI-based technique for obstacle detection and UAV flight control
Author: Salim Agharbi Ben Saddik
Degree: Bachelor in Aerospace Engineering
Academic year: Spring 2026

---

17. Licence and use

This repository is provided for academic reproducibility. Any use of the code with a real UAV must follow local safety rules, legal requirements and controlled testing procedures.
