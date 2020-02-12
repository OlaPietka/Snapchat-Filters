import argparse
import cv2
import numpy as np

#  Directory paths
CASCADE_DIR = "data/cascades/"
FILTER_DIR = "data/filters/"


def menage_filter(filter):
    if filter == "sunglasses":
        if len(eyes) != 2:
            return

        (x, y, min_w, min_h, max_w, max_h) = get_boundaries(eyes)

        offset_x = int(min_w * 0.6)
        offset_y = int(min_h * 0.3)

        sunglasses = process_image(filter, max_w + offset_x, min_h + offset_y)
        sunglasses = rotate_image(sunglasses, np.arctan(max_h / max_w) * 180 / np.pi - 20)

        overlay_image(y - int(offset_y / 2), x - int(offset_x / 2), sunglasses)
    elif filter == "santahat":
        for (x, y, w, h) in faces:
            offset_x = int(w * 0.4)
            offset_y = int(h * 0.2)

            santa_hat = process_image(filter, w + offset_x, h + offset_y)

            overlay_image(y - int(1.8*offset_y), x - int(0.8*offset_x), santa_hat)
    elif filter == "mask":
        for (x, y, w, h) in faces:
            offset_x = int(w * 0.2)
            offset_y = int(h * 0.2)

            mask = process_image(filter, w + offset_x, h + offset_y)

            overlay_image(y - int(0.2*offset_y), x - int(0.5*offset_x), mask)
    elif filter == "santabeard":
        for (x, y, w, h) in faces:
            offset_x = int(w * 0.2)
            offset_y = int(h * 0.2)

            santa_beard = process_image(filter, w + offset_x, h + offset_y)

            overlay_image(y + int(3.0 * offset_y), x - int(0.5 * offset_x), santa_beard)
    elif filter == "mustache":
        for (x, y, w, h) in faces:
            offset_x = int(w * -0.5)
            offset_y = int(h * -0.5)

            mustache = process_image(filter, w + offset_x, h + offset_y)

            overlay_image(y - int(1.0 * offset_y), x - int(0.5 * offset_x), mustache)
    elif filter == "eyeballs":
        for (x, y, w, h) in eyes:
            offset_x = int(w * -0.1)
            offset_y = int(h * -0.1)

            mustache = process_image(filter, w + offset_x, h + offset_y)

            overlay_image(y - int(1.0 * offset_y), x - int(0.5 * offset_x), mustache)


def process_image(name, w, h):
    image = cv2.imread(FILTER_DIR + name + ".png", -1)
    return cv2.resize(image, (w, h))


def overlay_image(offset_y, offset_x, image):
    filter_w, filter_h, _ = image.shape
    for i in range(0, filter_h):
        for j in range(0, filter_w):
            #  Check if image is outside of frame
            if i + offset_x >= frame.shape[1] or j + offset_y >= frame.shape[0]:
                break
            if image[j, i][3] != 0:
                frame[offset_y + j, offset_x + i] = image[j, i]


def get_boundaries(rects):
    if rects[0][0] > rects[1][0]:
        ((rx, ry, rw, rh), (lx, ly, lw, lh)) = rects
    else:
        ((lx, ly, lw, lh), (rx, ry, rw, rh)) = rects

    x = lx
    y = ly
    min_w = lw
    min_h = lh
    max_w = rx + rw - lx
    max_h = ly + lh - ry

    return x, y, min_w, min_h, max_w, max_h


def show_boundaries(rects, color):
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)


def rotate_image(mat, angle):
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    return cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))


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
    ap.add_argument("-f", "--filter", nargs='+', required=True, help="filters to use [santahat, mask, sunglasses]")
    ap.add_argument("-b", "--boundaries", required=False, action='store_true', help="show boundaries of detected areas (optional)")
    ap.add_argument("-ce", "--cascade_eye", required=False, help="haar cascade for eye detection (optional)")
    ap.add_argument("-cf", "--cascade_face", required=False, help="haar cascade for face detection (optional)")
    args = vars(ap.parse_args())

    #  Creating a face cascade
    face_cascade_path = CASCADE_DIR + (args["cascade_face"] if args["cascade_face"] else "face.xml")
    eye_cascade_path = CASCADE_DIR + (args["cascade_eye"] if args["cascade_eye"] else "eye.xml")

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

        #  Detect eyes and feces in our frame.
        eyes = detect_multi_scale(eye_cascade, 50, (10, 10))
        faces = detect_multi_scale(face_cascade, 50, (30, 30))

        #  Loop over a list of filters
        for filter in args["filter"]:
            menage_filter(filter)

        # Show boundaries if parameter selected
        if args["boundaries"]:
            show_boundaries(eyes, color=(255, 0, 0))
            show_boundaries(faces, color=(0, 255, 0))

        #  Display the resulting frame
        cv2.imshow('Snapchat filter', cv2.flip(frame, 1))

        # 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
