# Based on :
# https://raw.githubusercontent.com/Pavankunchala/Fitness-Tracking-App/main/gym_code.py
# https://github.com/Pavankunchala/Fitness-Tracking-App
# https://www.youtube.com/watch?v=bhoraBX2Dnk

# python gymcode.py -v .\Squats.mp4 -c 0 --det 0.3 --track 0.3
# python gymcode.py -v handstand.mp4 -c 0 --det 0.3 --track 0.3
# See also https://github.com/Pradnya1208/Squats-angle-detection-using-OpenCV-and-mediapipe_v1 for toelichting
# https://github.com/Pradnya1208/Squats-angle-detection-using-OpenCV-and-mediapipe_v1/blob/main/Squat%20pose%20estimation.ipynb

# https://google.github.io/mediapipe/solutions/pose.html
# https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
import cv2
import mediapipe as mp
import numpy as np
import argparse
import numpy as np
import time
import streamlit as st
from datetime import datetime
import tempfile

def run(run_streamlit, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate):
    line_color = (255, 255, 255)
    line_color_r = (255, 0, 0)  # used for right side
    line_color_g = (0, 255, 0)
    line_color_b = (0, 0, 255)

    start_time = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    def calculate_angle(a, b, c):
        a = np.array(a)  # first
        b = np.array(b)  # mid
        c = np.array(c)  # end

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=(line_color_g))
    drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=(line_color))

    if run_streamlit:

        # STREAMLIT
        # https://discuss.streamlit.io/t/how-to-access-uploaded-video-in-streamlit-by-open-cv/5831/6
        f = st.file_uploader("Upload file (mp4)", ['mp4'])
        
        if f is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            vid = cv2.VideoCapture(tfile.name)
        else:
            st.stop()

        stframe = st.empty()
    else:
        vid = cv2.VideoCapture(input_file)



    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    if output_file !=None:
        out = cv2.VideoWriter(output_file, codec, fps, (width, height))

    with mp_pose.Pose(
        min_detection_confidence=detection_confidence,         min_tracking_confidence=tracking_confidence,         model_complexity=complexity,         smooth_landmarks=True,     ) as pose:
        
        while vid.isOpened():
            success, image = vid.read()
            
            if not success:
                st.info("READY.")
            if rotate:
                image = cv2.rotate(image,cv2.ROTATE_180)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            image.flags.writeable = False
            results = pose.process(image)
            eyesVisible = False
            shoulderVisible = True

            # code for pose extraction

            landmarks = results.pose_landmarks.landmark

            # Check if both eyes are visible.

            left_eye = [
                landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y,             ]

            right_eye = [
                landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y,             ]

            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,             ]
            shoulder_r = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,             ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,             ]
            elbow_r = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,             ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,             ]
            wrist_r = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,             ]

            nose = [
                landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y,             ]

            # Get Tje Corridnates of Hip
            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,             ]
            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,             ]
            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,             ]
            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,             ]
            left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,             ]
            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,             ]

            # Put the Values for visibility

            # visiblity for Eyes
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0

            # fOR NOSE
            landmarks[mp_pose.PoseLandmark.NOSE.value].visibility = 0

            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0

            # fOR eAR
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

            # print('LeftEye',left_visible)

            # Check if both shoulders are visible.
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y,]

            # Midpointts
            midpoint_shoulder_x = (int(shoulder[0] * image_width) + int(shoulder_r[0] * image_width)) / 2
            midpoint_shoulder_y = (int(shoulder[1] * image_height) + int(shoulder_r[1] * image_height)) / 2

            midpoint_hip_x = (int(left_hip[0] * image_width) + int(right_hip[0] * image_width)) / 2
            midpoint_hip_y = (int(left_hip[1] * image_height) + int(right_hip[1] * image_height)) / 2

            based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x) / 2)
            based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y) / 2)
            base_mid = [based_mid_x, based_mid_y]
            neck_point_x = (int(nose[0] * image_width) + int(midpoint_shoulder_x)) / 2
            neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y)) / 2


            # angles
            left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
            right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
            left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))

            right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))

            left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

            # HANDSTAND

            left_shoulder_angle = int(calculate_angle(left_hip, shoulder, elbow))
            right_shoulder_angle = int(calculate_angle(right_hip, shoulder_r, elbow_r))

            left_hip_angle = int(calculate_angle(shoulder, left_hip, left_knee))
            right_hip_angle = int(calculate_angle(shoulder_r, right_hip, right_knee))

            # ppm = 10.8

            # left_arm_motion = left_arm_angle* left_arm_length

            # left_arm_motion = left_arm_motion/ppm

            # newpoint_left = [left_hip[0] +5,right_hip[0] +5]

            mid_point_x = (int(left_hip[0] * image_width) + int(right_hip[0] * image_width)) / 2
            mid_point_y = (int(left_hip[1] * image_height) + int(right_hip[1] * image_height)) / 2

            # cv2.circle(image,(int(mid_point_x) ,int(mid_point_y +30 )),15,(0,255,255),-1)

            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0

            # cv2.line(            #     image,             #     (int(shoulder[0] * image_width), int(shoulder[1] * image_height)),             #     (int(neck_point_x), int(neck_point_y)),             #     (line_color),             #     3,             # )

            # cv2.line(            #     image,             #     (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)),             #     (int(neck_point_x), int(neck_point_y)),             #     (line_color_r),             #     3,             # )

            cv2.line(image, (int(left_ankle[0] * image_width), int(left_ankle[1] * image_height)), (int(left_knee[0] * image_width), int(left_knee[1] * image_height)), (line_color), 3,             )

            # THESE LINES ARE NOT SHOWN #TOFIX
            cv2.line(image, (int(right_ankle[0] * image_width), int(right_ankle[1] * image_height)), (int(right_knee[0] * image_width), int(right_knee[1] * image_height)), (line_color_r), 3,             )

            cv2.line(image, (int(left_hip[0] * image_width), int(left_hip[1] * image_height)), (int(left_knee[0] * image_width), int(left_knee[1] * image_height)), (line_color), 3,             )

            cv2.line(image, (int(right_hip[0] * image_width), int(right_hip[1] * image_height)), (int(right_knee[0] * image_width), int(right_knee[1] * image_height)), (line_color_r), 3,             )

            cv2.line(image, (int(wrist[0] * image_width), int(wrist[1] * image_height)), (int(elbow[0] * image_width), int(elbow[1] * image_height)), (line_color), 3,             )

            # this one doesnt work #TOFIX
            cv2.line(image, (int(wrist_r[0] * image_width), int(wrist_r[1] * image_height)), (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), (line_color_b), 3,             )

            cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), (int(elbow[0] * image_width), int(elbow[1] * image_height)), (line_color), 3,             )
            cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), (line_color_r), 3,             )

            # # shoulder   to mid point HANDSTAND
            # cv2.line(image,             #     (int(midpoint_shoulder_x), int(midpoint_shoulder_y)),             #     (int(based_mid_x), int(based_mid_y)),             #     (line_color),             #     3,             #     cv2.LINE_4,             # )

            # # neck to mid point
            # cv2.line(image,             #     (int(neck_point_x), int(neck_point_y)),             #     (int(based_mid_x), int(based_mid_y)),             #     (line_color),             #     3,             #     cv2.LINE_4,             # )

            # # mid to hips
            # cv2.line(image,             #     (int(based_mid_x), int(based_mid_y)),             #     (int(left_hip[0] * image_width), (int(left_hip[1] * image_height))),             #     (line_color),             #     3,             #     cv2.LINE_8,             # )

            # cv2.line(image,             #     (int(based_mid_x), int(based_mid_y)),             #     (int(right_hip[0] * image_width), (int(right_hip[1] * image_height))),             #     (line_color_r),             #     3,             #     cv2.LINE_8, 
            # shouder to hips
            cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), (int(left_hip[0] * image_width), (int(left_hip[1] * image_height))), (line_color), 3, cv2.LINE_8,             )

            cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), (int(right_hip[0] * image_width), (int(right_hip[1] * image_height))), (line_color_r), 3, cv2.LINE_8,             )

            ##neck point

            # cv2.circle(image, (int(neck_point_x), int(neck_point_y)), 4, (line_color), 5
            # )

            # create new circles at that place
            cv2.circle(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 4, (line_color), 3,             )
            cv2.circle(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 4, (line_color), 3,             )
            # mid point
            # cv2.circle(image, (int(based_mid_x), int(based_mid_y)), 4, (line_color), 5)
            legend = False
            if legend:
                cv2.rectangle(
                    image, (image_width, 0), (image_width - 300, 350), (0, 0, 0), -1
                )
                cv2.putText(
                    image,     "Angles",     (image_width - 300, 30),     cv2.FONT_HERSHEY_COMPLEX,     1,     (0, 255, 255),     2, )

                # HANDSTAND SPECIFIC
                cv2.putText(
                    image,     "Left Shoulder Angle: " + str(left_shoulder_angle),     (image_width - 290, 70),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )
                cv2.putText(
                    image,     "Right Shoulder Angle: " + str(right_shoulder_angle),     (image_width - 290, 110),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )
                cv2.putText(
                    image,     "Left Elbow Angle: " + str(left_arm_angle),     (image_width - 290, 150),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )
                cv2.putText(
                    image,     "Right Elbow Angle: " + str(right_arm_angle),     (image_width - 290, 190),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )

                # HANDSTAND SPECIFIC
                cv2.putText(
                    image,     "Left Hip Angle: " + str(left_hip_angle),     (image_width - 290, 230),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )
                cv2.putText(
                    image,     "Right Hip Angle: " + str(right_hip_angle),     (image_width - 290, 270),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )

                cv2.putText(
                    image,     "Left Knee Angle: " + str(left_leg_angle),     (image_width - 290, 310),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )
                cv2.putText(
                    image,     "Right Knee Angle: " + str(right_leg_angle),     (image_width - 290, 340),     cv2.FONT_HERSHEY_COMPLEX,     0.7,     (0, 255, 255),     2,     cv2.LINE_AA, )

            # cv2.putText(image, 'Left arm motion: ' + str(left_arm_motion), (image_width-290, 230), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # print('left eye',left_eye)

            # print('Is Eye Visible',eyesVisible)
            # print('Is Shoulder Visible',shoulderVisible)

            # cv2.putText(image,"left elbow" + str(left_arm_angle),(int(image_width - 250),int(40)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            # writing angles
            cv2.putText(
                image, f"knee: {str(left_leg_angle)} / {str(right_leg_angle)}", (int(left_knee[0] * image_width - 40), int(left_knee[1] * image_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 244), 2, cv2.LINE_AA,             )

            cv2.putText(
                image, f"hip: {str(left_hip_angle)} / {str(right_hip_angle)}", (int(left_hip[0] * image_width - 40), int(left_hip[1] * image_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 244), 2, cv2.LINE_AA,             )

            cv2.putText(
                image, f"elbow: {str(left_arm_angle)} / {str(right_arm_angle)}", (int(elbow_r[0] * image_width - 40), int(elbow_r[1] * image_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 244), 2, cv2.LINE_AA,             )
            cv2.putText(
                image, f"shoulder: {str(left_shoulder_angle)} / {str(right_shoulder_angle)}", (int(shoulder[0] * image_width - 40), int(shoulder[1] * image_height)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 244), 2, cv2.LINE_AA,             )

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, drawing_spec_points, connection_drawing_spec=drawing_spec,             )

            fps = 1.0 / (time.time() - start_time)

            final_frame = image
            if output_file!=None:
                out.write(final_frame)

            final_frame = cv2.resize(final_frame, (0, 0), fx=0.4, fy=0.4)
            if run_streamlit:
                stframe.image(final_frame)
            else:
                cv2.imshow("Pose", final_frame)

                if cv2.waitKey(1) & 0xFF == 27:

                    break
        if not run_streamlit:
            vid.release()
            out.release()
            cv2.destroyAllWindows()
    run(run_streamlit, input_file, detection_confidence, tracking_confidence, complexity, rotate)
    

def check_streamlit():
    """
    Function to check whether python code is run within streamlit
    https://discuss.streamlit.io/t/how-to-check-if-code-is-run-inside-streamlit-and-not-e-g-ipython/23439/8
    Returns
    -------
    use_streamlit : boolean
        True if code is run within streamlit, else False
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit

def main():
    
    run_streamlit = check_streamlit()
    if run_streamlit:
        detection_confidence = st.sidebar.number_input("Detection confidence",0.0,1.0,0.5) 
        tracking_confidence =  st.sidebar.number_input("Tracking confidence",0.0,1.0,0.5) 
        complexity = st.sidebar.selectbox("Complexity", [0,1,2], index=1)
        rotate = st.sidebar.selectbox("Rotate", [True,False], index=1)
        input_file = None
    else:
        detection_confidence = 0.2 # args.det
        tracking_confidence = 0.2 #args.track
        complexity = 0 # 1 #  args.complexity
        rotate=True
        input_file = "theo.mp4"
        
    now = datetime.now() # current date and time
    date_time_now = now.strftime("%Y%m%d_%H%M%S")
    output_file = None # f"handstand_out_{date_time_now}.mp4"

    run(run_streamlit, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate)

if __name__=='__main__':
    main()