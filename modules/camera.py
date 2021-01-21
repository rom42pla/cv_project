from os import makedirs
from os.path import exists, join

from threading import Thread
import cv2
import time
import numpy as np

import torch
import torchvision.transforms as transforms

class Camera:

    def __init__(self, assets_path: str,
                 cam_number: int = 0, resolution: int = 720, show_fps: bool = True,
                 n_frames: int = 16,
                 seconds_to_be_recorded: float = 2,
                 window_size: int = 175,
                 window_color: tuple = (0, 0, 255),
                 window_name: str = 'ASL Recognizer'):
        # eventually creates output directory
        self.assets_path = assets_path
        self.samples_path, self.letters_examples = join(assets_path, "samples"), join(assets_path, "letters_examples")
        if not exists(self.samples_path):
            makedirs(self.samples_path)
        if not exists(self.letters_examples):
            makedirs(self.letters_examples)

        # sets camera's properties
        self.is_running = False
        self.vid, self.thread = cv2.VideoCapture(cam_number), None

        # states' variables
        self.state_starting_time = None
        self.seconds_to_be_recorded = seconds_to_be_recorded
        self.saved_frames = []
        self.asl_recognizer = None
        self.window_name = window_name

        # sets the resolution of the webcam
        assert isinstance(resolution, int) or isinstance(resolution, tuple) or isinstance(resolution, list)
        if isinstance(resolution, int):
            resolution = (int((resolution * 16) / 9), resolution)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        _, initial_frame = self.vid.read()
        self.resolution = initial_frame.shape[1], initial_frame.shape[0]

        # whether or not to show FPS label
        assert isinstance(show_fps, bool)
        self.show_fps = show_fps

        # settings about the recording square window
        assert window_size < self.resolution[0] // 2
        self.window_center = (self.resolution[0] // 2, self.resolution[1] // 2)
        self.window_color = window_color

        # structures used to save the various frames
        self.n_frames, self.window_size = n_frames, window_size

    def capture_frame(self):
        prev_frame_time, new_frame_time = 0, 0
        self.state_starting_time = time.time()

        while self.is_running:
            try:
                ret, frame = self.vid.read()
                show_frame, save_frame = self.frame_elaboration(frame,
                                                                horizontal_flip=True)
            except Exception as exception:
                print(exception)
                break

            if len(self.saved_frames) >= self.seconds_to_be_recorded * self.n_frames:
                self.saved_frames.pop(0)
            self.saved_frames += [cv2.cvtColor(save_frame, cv2.COLOR_BGR2RGB)]

            # eventually shows FPS on the app
            if self.show_fps:
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(img=show_frame, text=f"FPS: {np.round(fps)}",
                            org=(0, 30), color=(0, 255, 0),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
            cv2.imshow(self.window_name, show_frame)

            pressed_key = cv2.waitKey(1)
            if pressed_key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) == 0:
                self.stop()

    def frame_elaboration(self, save_frame, horizontal_flip: bool = True):
        # horizontally flips the image
        if horizontal_flip:
            save_frame = np.flip(save_frame, axis=1)

        show_frame = np.copy(save_frame)

        # draws a rectangle on the app
        show_frame = cv2.rectangle(show_frame,
                                   (self.window_center[0] - self.window_size, self.window_center[1] - self.window_size),
                                   (self.window_center[0] + self.window_size, self.window_center[1] + self.window_size),
                                   color=self.window_color, thickness=2)

        # commands label
        cv2.putText(img=show_frame, text=f"ESC - quit",
                    org=(0 + 20, self.resolution[1] - 20),
                    color=(255, 255, 255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)

        # show a progress bar
        percentage = (time.time() - self.asl_recognizer.waiting_since) / self.asl_recognizer.predictions_delta
        show_frame = cv2.rectangle(show_frame,
                                   (self.window_center[0] - self.window_size,
                                    self.window_center[1] - self.window_size),
                                   (self.window_center[0] - self.window_size + int(
                                       self.window_size * percentage * 2),
                                    25 + self.window_size),
                                   color=self.window_color, thickness=-1)

        # upper label
        if self.asl_recognizer:
            cv2.putText(img=show_frame, text="".join(self.asl_recognizer.predicted_letters[-15:]),
                        org=(self.window_center[0] - self.window_size, self.window_center[1] - self.window_size - 20),
                        color=self.window_color,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.25, thickness=1)

        # crops the area in the rectangle
        save_frame = save_frame[self.window_center[1] - self.window_size: self.window_center[1] + self.window_size,
                     self.window_center[0] - self.window_size: self.window_center[0] + self.window_size]

        show_frame = np.array(show_frame)

        return show_frame, save_frame

    def start(self):
        if self.is_running:
            raise Exception(f"Camera is already open")
        # starts the recording thread
        self.is_running = True
        self.thread = Thread(target=self.capture_frame)
        self.thread.start()

    def stop(self):
        # stops the recording thread
        self.is_running = False
        self.vid.release()
        cv2.destroyAllWindows()

    def get_saved_frames(self, frames_per_video: int = None):
        video = torch.from_numpy(np.array(self.saved_frames)).permute(0, 3, 1, 2) / 255
        if frames_per_video:
            video = video[np.linspace(start=0, stop=video.shape[0], num=frames_per_video, endpoint=False, dtype=int)]
        # preprocess the video
        new_video = torch.zeros(size=(video.shape[0], video.shape[1], 224, 224))
        for i_frame, frame in enumerate(video):
            transformations = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=224),
                transforms.ToTensor()
            ])
            new_video[i_frame] = transformations(frame)
        video = new_video
        from modules.utils import show_img
        show_img(video)
        return video
