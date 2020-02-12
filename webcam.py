import argparse
import cv2
import numpy as np

#  Directory paths
CASCADE_DIR = "data/cascades/"
FILTER_DIR = "data/filters/"


def get_image(filter, faces_rects, eyes_rects, frame):
    if filter == "sunglasses":
        if len(eyes_rects) is not 2:
            return

        if eyes_rects[0][0] > eyes_rects[1][0]:
            ((rx, ry, rw, rh), (lx, ly, lw, lh)) = eyes_rects
        else:
            ((lx, ly, lw, lh), (rx, ry, rw, rh)) = eyes_rects

        offset_x = int(lw * 0.6)
        offset_y = int(lh * 0.3)

        glasses = cv2.imread(FILTER_DIR + "sunglasses.png", -1)
        glasses = cv2.resize(glasses, (rx + rw - lx + offset_x, lh + offset_y))
        glasses = rotate_image(glasses, np.arctan((ly + lh - ry) / (rx + rw - lx)) * 180 / np.pi - 20)

        return overlay_image(ly - int(offset_y / 2), lx - int(offset_x / 2), glasses, frame)
    elif filter == "santahat":
        for (x, y, w, h) in faces_rects:
            offset_x = int(w * 0.4)
            offset_y = int(h * 0.2)

            santahat = cv2.imread(FILTER_DIR + "santahat.png", -1)
            santahat = cv2.resize(santahat, (w + offset_x, h + offset_y))

            frame = overlay_image(y - int(1.8*offset_y), x - int(0.8*offset_x), santahat, frame)
        return frame
    elif filter == "mask":
        for (x, y, w, h) in faces_rects:
            offset_x = int(w * 0.2)
            offset_y = int(h * 0.2)

            mask = cv2.imread(FILTER_DIR + "mask.png", -1)
            mask = cv2.resize(mask, (w + offset_x, h + offset_y))

            frame = overlay_image(y - int(0.2*offset_y), x - int(0.5*offset_x), mask, frame)
        return frame
    elif filter == "swap":
        if len(faces_rects) is not 2:
            return
        ((lx, ly, lw, lh), (rx, ry, rw, rh)) = faces_rects
        for i in range(0, lh):
            for j in range(0, lw):
                frame[ly + i, lx + j] = frame[ry + i, rx + j]
        return frame


def overlay_image(offset_y, offset_x, filter, frame):
    filter_w, filter_h, _ = filter.shape
    for i in range(0, filter_h):
        for j in range(0, filter_w):
            if filter[j, i][3] != 0:
                frame[offset_y + j, offset_x + i] = filter[j, i]
    return frame


def rotate_image(mat, angle):
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filter", nargs='+', required=True, help="filter to use [santahat, mask, sunglasses]")
    ap.add_argument("-ce", "--cascade_eye", required=False, help="haar cascade for eye detection (optional)")
    ap.add_argument("-cf", "--cascade_face", required=False, help="haar cascade for face detection (optional)")
    args = vars(ap.parse_args())

    #  Creating a face cascade
    face_cascade_path = CASCADE_DIR + (args["cascade_face"] if args["cascade_face"] else "face.xml")
    eye_cascade_path = CASCADE_DIR + (args["cascade_eye"] if args["cascade_eye"] else "eye.xml")

    print(face_cascade_path)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    #  Sets the video source to the default webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        #  Read one frame from the video source (webcam)
        _, frame = video_capture.read()

        #  Converted our webcam feed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        #  Detect faces in our frame.
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        #  Detect eyes in our frame.
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        #  Loop over a list of filters
        for filter in args["filter"]:
            filter = get_image(filter, faces, eyes, frame)


        #  Display the resulting frame
        cv2.imshow('FaceDetection', frame)

        # 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
