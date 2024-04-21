#                                   *****************ARUCO TRACKING FINAL*************************

#        WHAT CAN THE CODE DO ?     

#   Given an Aruco marker identifies its coordinates and tracks them with respect to the video feed window. The coordinates of the video stream
#   The coordinate system origin is top left corner and has a max range of (800,600).
#   This can be changed according to the requirement.
#   Can track multiple Arucos and associate each Aruco's coordinate with its own ID. 


#       REQUIREMENTS(Atleast What I used)

#       Opencv - 4.9.0
#       contrib - 4.9.0.80
#       Python - 3.12.2
#       for Aruco Tags : https://chev.me/arucogen/
#       for new syntax : https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get


#       DIRECTIONS FOR USE

#       from aruco_detector import findArucoMarkers, detect_markers
#       Just import it and use it accordingly 


import cv2
import cv2.aruco as aruco
import numpy as np

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Change the markerSize and totalMarkers depending on the dictionary being used. DICT_NxN_M. N and M correspond to markerSize and total no. of Markers
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    dictionary = cv2.aruco.getPredefinedDictionary(key)

    # These lines are as per the new syntax. To be followed from version 4.7+
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)

    centers = {}
    count = 0
    
    if draw:
        aruco.drawDetectedMarkers(img, markerCorners)
        if markerIds is not None:
            for i, markerCorner in enumerate(markerCorners):
                count =  count+1
                corners = markerCorner.reshape((4, 2)).astype(int)

                cx, cy = np.mean(corners, axis=0).astype(int)
                center = (cx, cy)

                cv2.circle(img, center, 1, (0, 0, 255), -1)

                marker_id = markerIds[i][0]
                centers[marker_id] = center

    return img, centers , count

def detect_markers():
    # 0 as susing default webcam
    cap = cv2.VideoCapture(0)

    while True:
        # success is a boolean and img hold the cam feed
        success, img = cap.read()
        img, centers , count = findArucoMarkers(img)

        # Dimensions of the video feed dialog box
        window_width = 800
        window_height = 600

        resized_img = cv2.resize(img, (window_width, window_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Resized Image", resized_img)

        if centers:
            print("Centers:", centers,'\n')
            print("Total number of markers : ",count)

        # press 'q' to terminate the feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_markers()
