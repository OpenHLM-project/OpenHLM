# This file is modified from https://github.com/real-stanford/universal_manipulation_interface

import collections
import copy
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict, cast

import cv2
import numpy as np
import scipy.interpolate as si


class OpencvIntrDict(TypedDict):
    """OpenCV fisheye intrinsic parameters."""

    DIM: np.ndarray
    K: np.ndarray
    D: np.ndarray


class ArucoConfigDict(TypedDict):
    """Configuration for ArUco tag detection."""

    aruco_dict: cv2.aruco.Dictionary
    marker_size_map: dict[int, float]


class TagPoseDict(TypedDict):
    """Pose and corners of a detected ArUco tag."""

    rvec: np.ndarray
    tvec: np.ndarray
    corners: np.ndarray


class TagDetectionResultDict(TypedDict):
    """Detection result for a single frame."""

    frame_idx: int
    time: float
    tag_dict: dict[int, TagPoseDict]


# =================== intrinsics ===================


def parse_fisheye_intrinsics(json_data: dict[str, Any]) -> OpencvIntrDict:
    """Reads camera intrinsics from OpenCameraImuCalibration JSON format and converts to OpenCV format.

    Args:
        json_data: Dictionary containing camera intrinsics in OpenCameraImuCalibration format.

    Returns:
        A dictionary containing OpenCV intrinsics with keys 'DIM' (dimensions), 'K' (camera matrix), and 'D' (distortion coefficients).

    Example:
    {
        "final_reproj_error": 0.17053819312281043,
        "fps": 60.0,
        "image_height": 1080,
        "image_width": 1920,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": 1.0026582765352035,
            "focal_length": 420.56809123853304,
            "principal_pt_x": 959.857586309181,
            "principal_pt_y": 542.8155851051391,
            "radial_distortion_1": -0.011968137016185161,
            "radial_distortion_2": -0.03929790706019372,
            "radial_distortion_3": 0.018577224235396064,
            "radial_distortion_4": -0.005075629959840777,
            "skew": 0.0
        },
        "nr_calib_images": 129,
        "stabelized": false
    }
    """
    assert json_data["intrinsic_type"] == "FISHEYE"
    intr_data = json_data["intrinsics"]

    # img size
    h = json_data["image_height"]
    w = json_data["image_width"]

    # pinhole parameters
    f = intr_data["focal_length"]
    px = intr_data["principal_pt_x"]
    py = intr_data["principal_pt_y"]

    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data["radial_distortion_1"],
        intr_data["radial_distortion_2"],
        intr_data["radial_distortion_3"],
        intr_data["radial_distortion_4"],
    ]

    opencv_intr_dict: OpencvIntrDict = {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64),
        "D": np.array([kb8]).T,
    }
    return opencv_intr_dict


def convert_fisheye_intrinsics_resolution(
    opencv_intr_dict: OpencvIntrDict, target_resolution: tuple[int, int]
) -> OpencvIntrDict:
    """Converts fisheye intrinsics parameters to a different resolution.

    Assumes that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in the horizontal dimension.

    Args:
        opencv_intr_dict: Dictionary containing OpenCV camera intrinsics ('DIM', 'K', 'D').
        target_resolution: Target image resolution as (width, height).

    Returns:
        A new dictionary with updated intrinsics for the target resolution.
    """
    iw, ih = opencv_intr_dict["DIM"]
    iK = opencv_intr_dict["K"]
    ifx = iK[0, 0]
    ify = iK[1, 1]
    ipx = iK[0, 2]
    ipy = iK[1, 2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([[ofx, 0, opx], [0, ofy, opy], [0, 0, 1]], dtype=np.float64)

    out_intr_dict = cast(OpencvIntrDict, copy.deepcopy(opencv_intr_dict))
    out_intr_dict["DIM"] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict["K"] = oK
    return out_intr_dict


def crop_fisheye_intrinsics(
    opencv_intr_dict: OpencvIntrDict, crop_box: tuple[int, int, int, int]
) -> OpencvIntrDict:
    """Adjust fisheye intrinsics after cropping an image.

    Args:
        opencv_intr_dict: Dictionary containing OpenCV camera intrinsics.
        crop_box: Crop box `(x0, y0, x1, y1)` in pixel coordinates.

    Returns:
        A new intrinsics dictionary for the cropped image.
    """
    x0, y0, x1, y1 = crop_box
    out_intr_dict = cast(OpencvIntrDict, copy.deepcopy(opencv_intr_dict))
    out_intr_dict["DIM"] = np.array([x1 - x0, y1 - y0], dtype=np.int64)
    out_intr_dict["K"][0, 2] -= x0
    out_intr_dict["K"][1, 2] -= y0
    return out_intr_dict


class FisheyeRectConverter:
    """Converts fisheye images to rectilinear images."""

    def __init__(
        self,
        K: np.ndarray,
        D: np.ndarray,
        DIM: np.ndarray | tuple[int, int],
        out_size: tuple[int, int],
        out_fov: float,
    ):
        """Initializes the FisheyeRectConverter.

        Args:
            K: The camera intrinsic matrix (3x3).
            D: The distortion coefficients.
            DIM: The original image dimensions.
            out_size: The desired output image size (width, height).
            out_fov: The desired output vertical field of view in degrees.
        """
        out_size_np = np.array(out_size)
        # vertical fov
        out_f = (out_size_np[1] / 2) / np.tan(out_fov / 180 * np.pi / 2)
        out_K = np.array(
            [
                [out_f, 0, out_size_np[0] / 2],
                [0, out_f, out_size_np[1] / 2],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), out_K, tuple(map(int, out_size)), cv2.CV_16SC2
        )

        self.map1 = map1
        self.map2 = map2

    def forward(self, img: np.ndarray) -> np.ndarray:
        """Applies rectilinear projection to the input fisheye image.

        Args:
            img: The input fisheye image.

        Returns:
            The undistorted rectilinear image.
        """
        rect_img = cv2.remap(
            img,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return rect_img


# ================= ArUcO tag =====================
def parse_aruco_config(aruco_config_dict: dict[str, Any]) -> ArucoConfigDict:
    """Parses ArUco configuration dictionary.

    Args:
        aruco_config_dict: Dictionary containing ArUco config ('aruco_dict', 'marker_size_map').

    Returns:
        A dictionary containing the initialized cv2.aruco.Dictionary and the marker size map.

    Example:
    aruco_dict:
        predefined: DICT_4X4_50
    marker_size_map: # all unit in meters
        default: 0.15
        12: 0.2
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict["aruco_dict"])

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict["marker_size_map"]
    default_size = marker_size_map.get("default", None)

    out_marker_size_map = dict()
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map:
            size = marker_size_map[marker_id]
        if size is not None:
            out_marker_size_map[marker_id] = size

    result: ArucoConfigDict = {
        "aruco_dict": aruco_dict,
        "marker_size_map": out_marker_size_map,
    }
    return result


def get_aruco_dict(predefined: str) -> cv2.aruco.Dictionary:
    """Gets a predefined ArUco dictionary by name.

    Args:
        predefined: The name of the predefined ArUco dictionary (e.g., 'DICT_4X4_50').

    Returns:
        The cv2.aruco.Dictionary object.
    """
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))


def detect_localize_aruco_tags(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: dict[int, float],
    fisheye_intr_dict: OpencvIntrDict,
    pixel_offset: tuple[int, int] = (0, 0),
    refine_subpix: bool = True,
) -> dict[int, TagPoseDict]:
    """Detects and localizes ArUco tags in a fisheye image.

    Args:
        img: The input image as a NumPy array.
        aruco_dict: The ArUco dictionary to use for detection.
        marker_size_map: Mapping from marker ID to marker physical size in meters.
        fisheye_intr_dict: Camera intrinsics dictionary with 'K' and 'D'.
        pixel_offset: Offset `(x0, y0)` of the image within the original frame.
        refine_subpix: Whether to use subpixel corner refinement. Defaults to True.

    Returns:
        A dictionary mapping marker ID to its pose and corners ('rvec', 'tvec', 'corners').
    """
    K = fisheye_intr_dict["K"]
    D = fisheye_intr_dict["D"]
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, rejectedImgPoints = detector.detectMarkers(img)

    if not corners or ids is None:
        return dict()

    tag_dict: dict[int, TagPoseDict] = dict()
    for this_id_arr, this_corners in zip(ids, corners, strict=False):
        this_id = int(this_id_arr[0])
        if this_id not in marker_size_map:
            continue

        marker_size_m = marker_size_map[this_id]
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)

        marker_points = np.array(
            [
                [-marker_size_m / 2, marker_size_m / 2, 0],
                [marker_size_m / 2, marker_size_m / 2, 0],
                [marker_size_m / 2, -marker_size_m / 2, 0],
                [-marker_size_m / 2, -marker_size_m / 2, 0],
            ],
            dtype=np.float32,
        )
        success, rvec, tvec = cv2.solvePnP(
            marker_points,
            undistorted,
            K,
            np.zeros((1, 5)),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            continue

        full_corners = this_corners.copy()
        full_corners[..., 0] += pixel_offset[0]
        full_corners[..., 1] += pixel_offset[1]

        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": full_corners.squeeze(),
        }
    return tag_dict


def draw_charuco_board(
    board: cv2.aruco.CharucoBoard, dpi: int = 300, padding_mm: float = 15
) -> np.ndarray:
    """Renders a ChArUco board image for printing.

    Args:
        board: The ChArUco board object.
        dpi: Dots per inch for the output image resolution.
        padding_mm: Padding around the board in millimeters.

    Returns:
        The rendered board image as a NumPy array.
    """
    grid_size = np.array(board.getChessboardSize())
    square_length_mm = board.getSquareLength() * 1000

    mm_per_inch = 25.4
    board_size_pixel = (
        (grid_size * square_length_mm + padding_mm * 2) / mm_per_inch * dpi
    )
    board_size_pixel = board_size_pixel.round().astype(np.int64)
    padding_pixel = int(padding_mm / mm_per_inch * dpi)
    board_img = board.generateImage(
        outSize=tuple(board_size_pixel), marginSize=padding_pixel
    )
    return board_img


def get_gripper_width(
    tag_dict: dict[int, TagPoseDict],
    left_id: int,
    right_id: int,
    nominal_z: float = 0.072,
    z_tolerance: float = 0.008,
) -> float | None:
    """Calculates gripper width based on detected ArUco tags on the fingers.

    Args:
        tag_dict: Dictionary of detected ArUco tags mapping ID to pose information.
        left_id: The ArUco tag ID on the left finger.
        right_id: The ArUco tag ID on the right finger.
        nominal_z: Expected nominal depth (Z) of the tags in meters.
        z_tolerance: Allowed tolerance around the nominal Z depth in meters.

    Returns:
        The calculated gripper width in meters, or None if detection fails or depth is out of bounds.
    """
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]["tvec"]
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width


# =========== image mask ====================


def canonical_to_pixel_coords(
    coords: list | np.ndarray, img_shape: tuple[int, int] = (2028, 2704)
) -> np.ndarray:
    """Converts normalized canonical coordinates to image pixel coordinates.

    Args:
        coords: Canonical coordinates in range [-0.5, 0.5] relative to image dimensions.
        img_shape: Tuple of (height, width) of the target image.

    Returns:
        Array of pixel coordinates.
    """
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts


def pixel_coords_to_canonical(
    pts: list | np.ndarray, img_shape: tuple[int, int] = (2028, 2704)
) -> np.ndarray:
    """Converts image pixel coordinates to normalized canonical coordinates.

    Args:
        pts: Image pixel coordinates.
        img_shape: Tuple of (height, width) of the target image.

    Returns:
        Array of canonical coordinates.
    """
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords


def draw_canonical_polygon(
    img: np.ndarray, coords: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    """Draws a polygon on an image using canonical coordinates.

    Args:
        img: The image to draw on.
        coords: Polygon vertices in canonical coordinates.
        color: The fill color as a BGR/RGB tuple.

    Returns:
        The modified image.
    """
    pts = canonical_to_pixel_coords(coords, tuple(img.shape[:2]))  # type: ignore
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, [pts], color=color)
    return img


def get_mirror_canonical_polygon() -> np.ndarray:
    """Returns canonical coordinates for masking the robot's mirror.

    Returns:
        Array of canonical coordinates representing the mirror mask polygons.
    """
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770],
    ]
    resolution = (2028, 2704)
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_mirror_crop_slices(
    img_shape: tuple[int, int] = (1080, 1920), left: bool = True
) -> tuple[slice, slice]:
    """Generates bounding box slices for cropping the mirror view from an image.

    Args:
        img_shape: Tuple of (height, width) of the target image.
        left: Whether to return the crop slices for the left or right mirror.

    Returns:
        A tuple of slices (height_slice, width_slice) for array indexing.
    """
    left_pts = [[290, 1120], [650, 1480]]
    resolution = (2028, 2704)
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    if not left:
        left_coords[:, 0] *= -1
    left_pts_pixel = canonical_to_pixel_coords(
        left_coords, img_shape=img_shape
    )
    left_pts_pixel = np.round(left_pts_pixel).astype(np.int32)
    slices = (
        slice(np.min(left_pts_pixel[:, 1]), np.max(left_pts_pixel[:, 1])),
        slice(np.min(left_pts_pixel[:, 0]), np.max(left_pts_pixel[:, 0])),
    )
    return slices


def get_gripper_canonical_polygon() -> np.ndarray:
    """Returns canonical coordinates for masking the default gripper.

    Returns:
        Array of canonical coordinates representing the gripper mask polygons.
    """
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704],
    ]
    resolution = (2028, 2704)
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_gripper_canonical_polygon_g1() -> np.ndarray:
    """Returns canonical coordinates for masking the G1 gripper.

    Returns:
        Array of canonical coordinates representing the G1 gripper mask polygons.
    """
    all_pts = [
        [1324, 1880],
        [971, 1817],
        [853, 1749],
        [588, 1593],
        [414, 1558],
        [0, 1552],
        [0, 2027],
        [2703, 2027],
        [2703, 1552],
        [2289, 1558],
        [2115, 1593],
        [1850, 1749],
        [1732, 1817],
        [1379, 1880],
    ]
    resolution = (2028, 2704)
    coords = pixel_coords_to_canonical(all_pts, resolution)
    return coords[None, ...]


def get_finger_canonical_polygon(
    height: float = 0.37, top_width: float = 0.25, bottom_width: float = 1.4
) -> np.ndarray:
    """Returns canonical coordinates for masking the fingers.

    Args:
        height: Height of the finger mask.
        top_width: Top width of the finger mask.
        bottom_width: Bottom width of the finger mask.

    Returns:
        Array of canonical coordinates representing the finger mask polygons.
    """
    # image size
    resolution = (2028, 2704)
    img_h, img_w = resolution

    # calculate coordinates
    top_y = 1.0 - height
    bottom_y = 1.0
    width = img_w / img_h
    middle_x = width / 2.0
    top_left_x = middle_x - top_width / 2.0
    top_right_x = middle_x + top_width / 2.0
    bottom_left_x = middle_x - bottom_width / 2.0
    bottom_right_x = middle_x + bottom_width / 2.0

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [
        [
            [bottom_left_x, bottom_y],
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bottom_right_x, bottom_y],
        ]
    ]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords


def draw_predefined_mask(
    img: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
    mirror: bool = True,
    gripper: bool = True,
    finger: bool = True,
    use_aa: bool = False,
    camera_setup: str = "default",
) -> np.ndarray:
    """Draws predefined masks (mirror, gripper, fingers) onto the image.

    Args:
        img: The image to mask.
        color: RGB/BGR tuple specifying the mask color. Defaults to black.
        mirror: Whether to draw the mirror mask.
        gripper: Whether to draw the gripper mask.
        finger: Whether to draw the finger mask.
        use_aa: Whether to use anti-aliased drawing.
        camera_setup: Camera setup string (e.g., 'default', 'g1').

    Returns:
        The image with applied masks.
    """
    if camera_setup == "g1":
        if mirror or finger:
            raise ValueError(
                "mirror or finger mask not supported for camera_setup='g1'"
            )
    all_coords = list()
    if mirror and camera_setup == "default":
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        if camera_setup == "g1":
            all_coords.extend(get_gripper_canonical_polygon_g1())
        else:
            all_coords.extend(get_gripper_canonical_polygon())
    if finger and camera_setup == "default":
        all_coords.extend(get_finger_canonical_polygon())
    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, tuple(img.shape[:2]))  # type: ignore
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img, [pts], color=color, lineType=flag)
    return img


def get_gripper_with_finger_mask(
    img: np.ndarray,
    height: float = 0.37,
    top_width: float = 0.25,
    bottom_width: float = 1.4,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Draws a simplified gripper and finger mask onto the image.

    Args:
        img: The image to mask.
        height: The relative height of the mask.
        top_width: The relative top width of the mask.
        bottom_width: The relative bottom width of the mask.
        color: The mask color as an RGB/BGR tuple.

    Returns:
        The masked image.
    """
    # image size
    img_h = img.shape[0]
    img_w = img.shape[1]

    # calculate coordinates
    top_y = 1.0 - height
    bottom_y = 1.0
    width = img_w / img_h
    middle_x = width / 2.0
    top_left_x = middle_x - top_width / 2.0
    top_right_x = middle_x + top_width / 2.0
    bottom_left_x = middle_x - bottom_width / 2.0
    bottom_right_x = middle_x + bottom_width / 2.0

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = np.array(
        [
            [
                [bottom_left_x, bottom_y],
                [top_left_x, top_y],
                [top_right_x, top_y],
                [bottom_right_x, bottom_y],
            ]
        ],
        dtype=np.int32,
    )

    img = cv2.fillPoly(img, [points[0]], color=color, lineType=cv2.LINE_AA)
    return img


def inpaint_tag(
    img: np.ndarray,
    corners: np.ndarray,
    tag_scale: float = 1.4,
    n_samples: int = 16,
) -> np.ndarray:
    """Inpaints an ArUco tag by filling it with the median background color.

    Args:
        img: The image containing the tag.
        corners: The detected corners of the ArUco tag.
        tag_scale: Scaling factor for the polygon used to extract background color.
        n_samples: Number of points to sample along the scaled boundary for median color estimation.

    Returns:
        The image with the inpainted tag.
    """
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center

    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0, 1, 2, 3, 4], list(scaled_corners) + [scaled_corners[0]], axis=0
    )(np.linspace(0, 4, n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:, 1], 0, img.shape[0] - 1),
        np.clip(sample_points[:, 0], 0, img.shape[1] - 1),
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)

    # draw tag with median color
    img = cv2.fillPoly(
        img, [scaled_corners.astype(np.int32)], color=median_color.tolist()
    )
    return img


# =========== other utils ====================


def get_image_transform(
    in_res: tuple[int, int],
    out_res: tuple[int, int],
    crop_ratio: float = 1.0,
    bgr_to_rgb: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a callable function that crops and resizes an image.

    Args:
        in_res: The input resolution as (width, height).
        out_res: The desired output resolution as (width, height).
        crop_ratio: The ratio of the input height to retain after cropping.
        bgr_to_rgb: Whether to convert BGR to RGB.

    Returns:
        A callable that transforms an input image to the desired properties.
    """
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray) -> np.ndarray:
        assert img.shape == (ih, iw, 3), (
            f"Expected shape {(ih, iw, 3)}, got {img.shape}"
        )
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img

    return transform


def detect_gripper_id(
    input: Path, nominal_z: float = 0.072, z_tolerance: float = 0.008
) -> int:
    """Detects the gripper hardware ID based on ArUco tag detection results.
    Args:
        input: Path to the pickle file containing tag detection results.
        nominal_z: Expected nominal depth (Z) of the tags in meters.
        z_tolerance: Allowed tolerance around the nominal Z depth in meters.

    Returns:
        The detected gripper ID as an integer.
    """
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance
    tag_detection_results: list[TagDetectionResultDict] = pickle.load(
        open(input, "rb")
    )

    # identify gripper hardware id
    n_frames = len(tag_detection_results)
    tag_counts = collections.defaultdict(lambda: 0)
    for frame in tag_detection_results:
        for key in frame["tag_dict"].keys():
            tvec = frame["tag_dict"][key]["tvec"]
            z = tvec[2]
            if zmin <= z <= zmax:
                tag_counts[key] += 1
    tag_stats = collections.defaultdict(lambda: 0.0)
    for k, v in tag_counts.items():
        tag_stats[k] = v / n_frames

    max_tag_id = np.max(list(tag_stats.keys()))
    tag_per_gripper = 6
    max_gripper_id = max_tag_id // tag_per_gripper

    gripper_prob_map = dict()
    for gripper_id in range(max_gripper_id + 1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        left_prob = tag_stats[left_id]
        right_prob = tag_stats[right_id]
        gripper_prob = min(left_prob, right_prob)
        if gripper_prob <= 0:
            continue
        gripper_prob_map[gripper_id] = gripper_prob
    if len(gripper_prob_map) == 0:
        raise ValueError(f"Cannot detect gripper id from tags in {input}")

    gripper_probs = sorted(gripper_prob_map.items(), key=lambda x: x[1])
    gripper_id = gripper_probs[-1][0]
    gripper_prob = gripper_probs[-1][1]

    return gripper_id
