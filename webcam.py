import argparse
import sys

import cv2
import numpy as np
import pygame
import filter

#  Directory paths
CASCADE_DIR = "data/cascades/"


def show_boundaries(rects, color):
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)


def detect_multi_scale(cascade, min_neighbors, min_size):
    return cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filter", nargs='+', required=False, help="filters to use")
    ap.add_argument("-p", "--package", required=False, help="filter package (optional)")
    ap.add_argument("-b", "--boundaries", required=False, action='store_true', help="show boundaries of detected areas (optional)")
    ap.add_argument("-ce", "--cascade_eye", required=False, help="haar cascade for eye detection (optional)")
    ap.add_argument("-cf", "--cascade_face", required=False, help="haar cascade for face detection (optional)")
    args = vars(ap.parse_args())

    #  Creating a face cascade
    face_cascade_path = CASCADE_DIR + (args["cascade_face"] if args["cascade_face"] else "face.xml")
    eye_cascade_path = CASCADE_DIR + (args["cascade_eye"] if args["cascade_eye"] else "eye.xml")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    #  Init display
    pygame.init()
    pygame.display.set_caption("Snapchat filters")
    screen = pygame.display.set_mode([640, 480])

    #  Sets the video source to the default webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        #  Read one frame from the video source (webcam)
        _, frame = video_capture.read()

        #  Converted our webcam feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        #  Detect eyes and feces in our frame.
        eyes = detect_multi_scale(eye_cascade, 50, (10, 10))
        faces = detect_multi_scale(face_cascade, 50, (30, 30))

        # Show boundaries
        if args["boundaries"]:
            show_boundaries(eyes, (255, 0, 0))
            show_boundaries(faces, (0, 255, 0))

        #  Mange screen
        screen.fill([0, 0, 0])
        frame = np.rot90(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        # Loop over the list of filters
        if args["filter"]:
            for filter_name in args["filter"]:
                filter.show_filter(filter_name, eyes, faces, screen)
        elif args["package"]:
            filter.show_filter_package(args["package"], eyes, faces, screen)

        #  Display the resulting frame
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                sys.exit(0)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    pygame.quit()
