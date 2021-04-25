import cv2
import os
from os.path import isfile, join
import numpy as np
import json
from scipy.spatial import distance as dist
from tqdm import tqdm


def sort_coords(array):
    sortval = array[:, :, 0] * 4 + array[:, :, 1]
    sortval *= -1
    sortedIdx = np.argsort(sortval)
    result = array[np.arange(len(array))[:, np.newaxis], sortedIdx]
    return result

left_column_coords = np.load('left_column.npy')
left_column_coords = sort_coords(left_column_coords)
right_column_coords = np.load('right_column.npy')
right_column_coords = sort_coords(right_column_coords)

with open('keypoints.json') as json_file:
    keypoints = json.load(json_file)

pathIn='Frames/'
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
files.sort(key=lambda x: x[5:-4])
files.sort()
frame_array = []

Frame = cv2.imread(pathIn+files[0], 1)
height, width, color = Frame.shape
print(Frame.shape)

frame_count = 0
distances = []
ratios = []
jumps = []
jump_count = 1

for p in tqdm(range(len(files))):

    filename = pathIn + files[p]
    frame = cv2.imread(filename, 1)

    keypoint_id = 0
    left_point_id = 0
    right_point_id = 0


    for n in range(len(files)):
        left_coords = left_column_coords[n]
        right_coords = right_column_coords[n]
        keypoints_for_frame = keypoints[n]
        points = keypoints_for_frame['keypoints']
        points = [point for point in points if point > 1]

        if frame_count == n:
            for coord_left, coord_right in zip(left_coords, right_coords):
                text_left = "{}".format(left_point_id)
                cv2.circle(frame, (int(coord_left[0]), int(coord_left[1])), 3, (0, 0, 0), -1)
                cv2.putText(frame, text_left, (int(coord_left[0]), int(coord_left[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1)

                text_right = "{}".format(right_point_id)
                cv2.circle(frame, (int(coord_right[0]), int(coord_right[1])), 3, (0, 0, 0), -1)
                cv2.putText(frame, text_right, (int(coord_right[0]), int(coord_right[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1)

                if(left_point_id == right_point_id == 2):
                    step_left_coord = (int(coord_left[0]), int(coord_left[1]))
                    step_right_coord = (int(coord_right[0]), int(coord_right[1]))
                    center_point_step = (step_right_coord[0]+(step_left_coord[0]-step_right_coord[0])/2, step_right_coord[1]+(step_left_coord[1]-step_right_coord[1])/2)
                    cv2.circle(frame, (int(center_point_step[0]), int(center_point_step[1])), 3, (0, 0, 0), -1)

                if (left_point_id == right_point_id == 1):
                    step_left_coord = (int(coord_left[0]), int(coord_left[1]))
                    step_right_coord = (int(coord_right[0]), int(coord_right[1]))
                    center_point_bottom = (step_right_coord[0] + (step_left_coord[0] - step_right_coord[0]) / 2,
                                         step_right_coord[1] + (step_left_coord[1] - step_right_coord[1]) / 2)
                    cv2.circle(frame, (int(center_point_bottom[0]), int(center_point_bottom[1])), 3, (0, 0, 0), -1)

                left_point_id += 1
                right_point_id += 1
            for x, y in zip(points[0::2], points[1::2]):
                text_point = "{}".format(keypoint_id)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(frame, text_point, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 0, 0), 1)

                if keypoint_id == 20:
                    leg_coord_1 = (int(x), int(y))
                if keypoint_id == 21:
                    leg_coord_2 = (int(x), int(y))
                if keypoint_id == 22:
                    leg_coord_3 = (int(x), int(y))
                if keypoint_id == 22:
                    leg_coord_4 = (int(x), int(y))
                keypoint_id += 1
        else:
            i = n
            left_column_coords = left_column_coords[i:]
            right_column_coords = right_column_coords[i:]
            keypoints = keypoints[i:]

            if frame is None:
                break
            break

    near_dist = dist.euclidean(leg_coord_1, center_point_step)
    known_euclid_dist = dist.euclidean(center_point_bottom, center_point_step)
    ratio = 50/known_euclid_dist
    ratios.append(ratio)
    if near_dist < 30:
        d_1 = dist.euclidean(leg_coord_1, center_point_bottom)
        d_2 = dist.euclidean(leg_coord_2, center_point_bottom)
        d_3 = dist.euclidean(leg_coord_3, center_point_bottom)
        d_4 = dist.euclidean(leg_coord_4, center_point_bottom)
        d_best = min(d_1, d_2, d_3, d_4)
        distances.append(d_best)
    elif len(distances) > 0:
        jump_height = np.mean(ratios) * max(distances)
        jumps.append(jump_height)
        distances = []

    frame_count = 0
    cv2.imshow('Output', frame)
    cv2.waitKey(2)

    if p == len(files)-2:
        break

print("Video is ready")
print("Information about jumps:")

for jump_id in jumps:
    print('Jump {}: the best result is {} cm'.format(jump_count, round(jump_id, 2)))
    jump_count += 1

cv2.destroyAllWindows()