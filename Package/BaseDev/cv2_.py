"""
This packet could be used as same as cv2.(of course, just some basic functions, others will be writen by yourself)
It solves -- method(s) of cv2 could not be called by dot. --
"""
import cv2
import numpy as np


class CV2:
    COLOR_BGR2HSV = cv2.COLOR_BGR2HSV
    COLOR_HSV2BGR = cv2.COLOR_HSV2BGR

    def __init__(self):
        pass

    @staticmethod
    def imread(file: str) -> np.ndarray:
        return cv2.imread(file)

    @staticmethod
    def imshow(window_name: str, img: np.ndarray):
        cv2.imshow(window_name, img)

    @staticmethod
    def imwrite(filename, img):
        cv2.imwrite(filename, img)

    @staticmethod
    def resize(img: np.ndarray, new_size: tuple) -> np.ndarray:
        return cv2.resize(img, new_size)

    @staticmethod
    def cvtColorToRGB(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def cvtColorToBGR(img: np.ndarray) -> np.ndarray:

        res = np.stack((img[:, :, 2], img[:, :, 1], img[:, :, 0]), axis=-1)
        return res

    @staticmethod
    def waitKey(t: int):
        cv2.waitKey(t)

    @staticmethod
    def rectangle(img: np.ndarray, start_point: tuple, end_point: tuple, color: tuple, thickness: int):
        image = cv2.rectangle(img, start_point, end_point, color, thickness)
        return image

    @staticmethod
    def putText(img: np.ndarray,
                text: str,
                org: tuple,
                font_face=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale: float = 0.5,
                color: tuple = (0, 0, 0),
                thickness: int = 1,
                line_type=cv2.LINE_AA,
                back_ground_color=None
                ):  # real signature unknown; restored from __doc__

        if back_ground_color is not None:
            wh, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
            w, h = wh[0], wh[1]
            top_left = (org[0], org[1] - h - baseline + 2)
            bottom_right = (org[0] + w, org[1]+baseline)

            CV2.rectangle(
                img,
                top_left,
                bottom_right,
                color=back_ground_color,
                thickness=-1
            )
        cv2.putText(img,
                    text,
                    org,
                    font_face,
                    font_scale,
                    color,
                    thickness,
                    line_type,
                    )

    @staticmethod
    def circle(
            img,
            center,
            radius,
            color,
            thickness=-1,
    ):
        cv2.circle(img, center, radius, color, thickness)

    @staticmethod
    def line(img, pt1, pt2, color, thickness):
        cv2.line(img, pt1, pt2, color, thickness)

    @staticmethod
    def flip(image, code: int) -> np.ndarray:
        assert code in [-1, 0, 1]
        return cv2.flip(image, code)

    @staticmethod
    def cvtColor(image, color_type):
        return cv2.cvtColor(image, color_type)


