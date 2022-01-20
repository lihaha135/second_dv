import numpy as np


def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
    """
    Estimate the velocity for an annotation.
    If possible, we compute the centered difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the velocity cannot be estimated, values are set to np.nan.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
    """

    current = self.get('sample_annotation', sample_annotation_token)

    has_prev = current['prev'] != ''
    has_next = current['next'] != ''

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.array([np.nan, np.nan, np.nan])

    if has_prev:
        first = self.get('sample_annotation', current['prev'])
    else:
        first = current

    first_box = Box(first['translation'], first['size'], Quaternion(first['rotation']),
                    name=first['category_name'], token=first['token'])
    sample = self.get('sample', first['sample_token'])
    sd_record = self.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

    if has_next:
        last = self.get('sample_annotation', current['next'])
    else:
        last = current

    last_box = Box(last['translation'], last['size'], Quaternion(last['rotation']),
                   name=last['category_name'], token=last['token'])
    sample1 = self.get('sample', last['sample_token'])
    sd_record1 = self.get('sample_data', sample1['data']["LIDAR_TOP"])
    cs_record1 = self.get('calibrated_sensor', sd_record1['calibrated_sensor_token'])
    pose_record1 = self.get('ego_pose', sd_record1['ego_pose_token'])

    # Move box to ego vehicle coord system
    first_box.translate(-np.array(pose_record['translation']))
    first_box.rotate(Quaternion(pose_record['rotation']).inverse)
    #  Move box to sensor coord system
    first_box.translate(-np.array(cs_record['translation']))
    first_box.rotate(Quaternion(cs_record['rotation']).inverse)

    last_box.translate(-np.array(pose_record1['translation']))
    last_box.rotate(Quaternion(pose_record1['rotation']).inverse)
    #  Move box to sensor coord system
    last_box.translate(-np.array(cs_record1['translation']))
    last_box.rotate(Quaternion(cs_record1['rotation']).inverse)

    # pos_last = np.array(last['translation'])
    # pos_first = np.array(first['translation'])

    pos_last = first_box.center
    pos_first = last_box.center

    pos_diff = pos_last - pos_first

    time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
    time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
    time_diff = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    if time_diff > max_time_diff:
        # If time_diff is too big, don't return an estimate.
        return np.array([np.nan, np.nan, np.nan])
    else:
        return pos_diff / time_diff