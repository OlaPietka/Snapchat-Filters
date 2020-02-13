import pygame

FILTER_DIR = "data/filters/"


def show_filter(filter_name, eyes, faces, screen):
    image = pygame.image.load(FILTER_DIR + filter_name + ".png")

    if "glasses" in filter_name:
        if len(eyes) != 2:
            return

        (x, y, min_w, min_h, max_w, max_h) = get_boundaries(eyes)
        process_image(image, 1.3*max_w, 1.2*min_h, x+1.18*max_w, y-0.1*max_h, screen)
    elif "mustache" in filter_name:
        for (x, y, w, h) in faces:
            process_image(image, 0.5*w, 0.5*h, x+0.75*w, y+0.44*h, screen)
    elif filter_name == "hat":
        for (x, y, w, h) in faces:
            process_image(image, 1.3*w, 1.2*h, x+1.15*w, y-0.7*h, screen)
    elif filter_name == "santahat":
        for (x, y, w, h) in faces:
            process_image(image, 1.4*w, 1.2*h, x+1.3*w, y-0.4*h, screen)
    elif filter_name == "mask":
        for (x, y, w, h) in faces:
            process_image(image, 1.2*w, 1.2*h, x+1.1*w, y-0.07*h, screen)
    elif filter_name == "santabeard":
        for (x, y, w, h) in faces:
            process_image(image, 1.1*w, 1.1*h, x+1.05*w, y+0.6*h, screen)
    elif filter_name == "eyeballs":
        for (x, y, w, h) in eyes:
            process_image(image, 0.9*w, 0.9*h, x+1.0*w, y, screen)
    elif filter_name == "ballmask":
        for (x, y, w, h) in faces:
            process_image(image, 0.9*w, 0.5*h, x+0.95*w, y+0.175*h, screen)
    elif filter_name == "clownhair":
        for (x, y, w, h) in faces:
            process_image(image, 1.8*w, 1.3*h, x+1.45*w, y-0.7*h, screen)
    elif filter_name == "clownnose":
        for (x, y, w, h) in faces:
            process_image(image, 0.3*w, 0.3*h, x+0.66*w, y+0.45*h, screen)


def show_gif():
    pass


def show_filter_package(package_name, eyes, faces, screen):
    if package_name == "santa":
        for f in ["santabeard", "santahat"]:
            show_filter(f, eyes, faces, screen)
    elif package_name == "clown":
        for f in ["clownhair", "clownnose"]:
            show_filter(f, eyes, faces, screen)
    elif package_name == "man":
        for f in ["mustache2", "hat", "glasses"]:
            show_filter(f, eyes, faces, screen)


def process_image(image, resize_x, resize_y, scale_x, scale_y, screen):
    image = pygame.transform.scale(image, (int(resize_x), int(resize_y)))
    screen.blit(image, (640 - int(scale_x), int(scale_y)))


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
    max_h = ly + lh - ry if ly + lh - ry > lh else ry + rh - ly

    return x, y, min_w, min_h, max_w, max_h
