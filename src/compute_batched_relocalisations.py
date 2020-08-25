#! /usr/bin/env python

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import transforms3d
from dual_quaternions import DualQuaternion

_logger = logging.getLogger(__name__)


def load_sequence(in_file: Path) -> List[np.ndarray]:
    """Load a sequence of poses from a text file.

    Each line in the file should look like this:
      {sequence-guid}/frame-{frame_idx} qw qx qy qz tx ty tz

    Frame IDs are expected in increasing order. If a frame_idx is missing,
    the returned list will contain a 4x4 NaN matrix in its place.

    :param in_file: A path to the text file containing poses.
    :return: A List[np.ndarray] of 4x4 pose matrices.
    """
    result = []

    with open(in_file, "r") as f:
        for line in f:
            chunks = line.split()
            assert len(chunks) == 8  # relative path + quat + trans
            guid = chunks[0].split("/")[0]
            frame_idx = int(chunks[0].split("-")[-1])
            assert guid == in_file.stem

            if len(result) < frame_idx:
                _logger.warning(
                    f"Frames {len(result)}-{frame_idx} are missing from "
                    f"{in_file}. Adding NaN poses instead."
                )
                while len(result) < frame_idx:
                    result.append(np.full((4, 4), np.nan))

            pose_quat_t = [float(x) for x in chunks[1:]]
            if np.isfinite(pose_quat_t).all():
                pose = np.eye(4)
                pose[0:3, 0:3] = transforms3d.quaternions.quat2mat(pose_quat_t[0:4])
                pose[0:3, 3] = pose_quat_t[4:7]
            else:
                pose = np.full((4, 4), np.nan)

            result.append(pose)

    return result


def save_clustered_results(
    clustered_poses: List[np.ndarray], out_base_folder: Path, seq_guid: str
) -> None:
    """Save a list of poses to a text file.

    The text file will have the following path: {out_base_folder}/{seq_guid}.txt
    Each line in the generated text file will look as follows:
      {seq_guid}/frame-{frame_idx} qw qx qy qz tx ty tz

    :param clustered_poses: A list of 4x4 pose matrices to save.
    :param out_base_folder: The base folder where to store the output files.
    :param seq_guid: GUID of the sequence.
    """
    out_file = out_base_folder / f"{seq_guid}.txt"
    _logger.info(f"Saving sequence results in {out_file}")
    with open(out_file, "w") as f:
        for frame_idx, pose in enumerate(clustered_poses):
            frame_tag = f"{seq_guid}/frame-{frame_idx:06d}"
            if np.isfinite(pose).all():
                quat = transforms3d.quaternions.mat2quat(pose[0:3, 0:3])
                trans = pose[0:3, 3]
                f.write(
                    f"{frame_tag} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
                    f"{trans[0]} {trans[1]} {trans[2]}\n"
                )
            else:
                f.write(f'{frame_tag} {"NaN " * 7}\n')


def transform_sequence(
    reloc_poses: Sequence[np.ndarray],
    gt_poses: Sequence[np.ndarray],
    reference_pose: np.ndarray,
) -> List[np.ndarray]:
    """Transform a list of relocalised poses (computed from different frames)
    as if they were all computed from a specific frame.

    The idea is that multiple relocalisation attempts from different frames
    (which are captured from different camera poses) should output consistent
    results. In this function we transform the relocalisation outputs using a
    known (and assumed good) camera trajectory.
    In theory, if all relocalisations were perfect and frame-to-frame camera
    tracking was accurate, the transformed relocalised poses would all be the same.
    Later we will exploit this idea and cluster the transformed poses, in order
    to output the best set of relocalisations for a batch of frames.

    :param reloc_poses: The poses output by a relocalisation algorithm.
        They are expected as 4x4 matrices describing camera-to-world transforms.
    :param gt_poses: The camera poses for the frames used to generate the reloc_poses above.
        They are expected as 4x4 matrices describing camera-to-world transforms.
        The origin of these poses does not have to coincide with the origin of
        the reloc_poses (and probably never will). We only care about the
        relative transforms between some frame and the reference frame below.
    :param reference_pose: The pose (as 4x4 matrix in camera-to-world, with the
        origin the same as the origin of gt_poses) of a frame for which we want
        to cluster the relocalisation outputs.
    :return: A List of relocalisation poses (camera-to-world, with the origin
    the same as the origin of reloc_poses) obtained transforming reloc_poses as
    if they were computed from reference_pose (its origin is the origin of the
    camera for gt_poses).
    """
    assert len(reloc_poses) == len(
        gt_poses
    ), "Reloc and GT pose lists must have the same length."
    transformed_sequence = []

    # reloc_poses are cam_t -> reloc_origin (wold)
    # gt_poses are cam_t -> current_origin (w)
    # reference_pose is cam_end -> current_origin (w)
    # we want to transform all reloc_poses into cam_end -> reloc_origin (wold)
    e_T_w = np.linalg.inv(reference_pose)
    for reloc_T_t, w_T_t in zip(reloc_poses, gt_poses):
        e_T_t = e_T_w @ w_T_t
        t_T_e = np.linalg.inv(e_T_t)
        reloc_T_e = reloc_T_t @ t_T_e

        transformed_sequence.append(reloc_T_e)

    return transformed_sequence


def angular_separation(r1: np.ndarray, r2: np.ndarray) -> float:
    """Compute the angular difference between two rotation matrices.

    :param r1: A rotation matrix.
    :param r2: A rotation matrix.
    :return: The angle (in degrees) of the angle-axis transform that maps
        r1 to r2.
    """
    # First compute the rotation that maps r1 to r2.
    dr = r2 @ r1.transpose()
    # Then extract the angle.
    _, angle = transforms3d.axangles.mat2axangle(dr)
    # Normalise the angle.
    if angle > np.pi:
        angle = 2 * np.pi - angle

    # Return the angle in degrees.
    return angle * 180 / np.pi


def translation_separation(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute the difference between two translations.

    :param t1: A translation vector.
    :param t2: A translation vector.
    :return: The difference between the two vectors.
    """
    return np.linalg.norm(t1 - t2)


def pose_close(
    p1: np.ndarray, p2: np.ndarray, angle_thresh=5.0, trans_thresh=0.05
) -> bool:
    """Determine whether two poses are close to each other (according to some
    threshold on the angular and translational components)

    :param p1: A 4x4 camera pose.
    :param p2: A 4x4 camera pose
    :param angle_thresh: Maximum threshold in degrees to consider the two poses close to each other.
    :param trans_thresh: Maximum threshold in metres to consider the two poses close to each other.
    :return: Whether the two poses are close to each other.
    """
    return (
        angular_separation(p1[0:3, 0:3], p2[0:3, 0:3]) < angle_thresh
        and translation_separation(p1[0:3, 3], p2[0:3, 3]) < trans_thresh
    )


def blend_poses(poses: Sequence[np.ndarray]) -> np.ndarray:
    """Blend a list of 4x4 transformation matrices together using dual
    quaternion blending.

    See: https://www.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf

    :param poses: A list of poses to blend.
    :return: The result of DQB applied on the input poses.
    """
    dq = DualQuaternion.from_dq_array([0, 0, 0, 0, 0, 0, 0])

    # We use a constant weight for all poses.
    weight = 1.0 / len(poses)

    for p in poses:
        dq += weight * DualQuaternion.from_homogeneous_matrix(p)
    dq.normalize()

    return dq.homogeneous_matrix()


@dataclass
class Cluster:
    centroid: Optional[np.ndarray] = None
    elements: List[np.ndarray] = field(default_factory=list)


def find_closest_cluster(
    pose: np.ndarray, clusters: Sequence[Cluster]
) -> Optional[int]:
    """Find a cluster close enough to the specified pose.

    Specifically, iterated over each individual pose in each cluster, and if
    one pose is deemed close to the input pose then the corresponding cluster
    index is returned.

    :param pose: A 4x4 camera pose.
    :param clusters: A list of Clusters.
    :return: The index of the first cluster containing a pose "close" to the
        input pose, or None otherwise.
    """
    # Maybe later we might want to check wrt. the centroid instead of all the
    # elements, depending on the performances.
    for cluster_idx, cluster in enumerate(clusters):
        for clustered_pose in cluster.elements:
            if pose_close(pose, clustered_pose):
                # Just assign the first cluster that satisfies the reqs.
                return cluster_idx

    return None


def cluster_poses(poses: Sequence[np.ndarray]) -> List[Cluster]:
    """Process a sequence of relocalised poses and create clusters.

    The clustering algorithm is naive: assign each input pose to the first
    cluster close enough to it. If no cluster is close enough, initialise a
    new cluster with the pose.

    :param poses: A list of relocalised poses to cluster.
    :return: A List of clusters created from the poses. Sorted from the largest
        cluster (in number of poses) to the smallest.
    """
    clusters: List[Cluster] = []
    for pose_idx, pose in enumerate(poses):
        # Find cluster
        cluster_idx = find_closest_cluster(pose, clusters)

        if cluster_idx is not None:
            _logger.debug(f"Assigning pose {pose_idx} to cluster {cluster_idx}")
            clusters[cluster_idx].elements.append(pose)
        else:
            _logger.debug(f"Assigning pose {pose_idx} to a new cluster.")
            cluster = Cluster()
            cluster.elements.append(pose)
            clusters.append(cluster)
    _logger.debug(f"Created {len(clusters)} clusters.")

    # Compute centroids.
    for c in clusters:
        c.centroid = blend_poses(c.elements)

    # Sort by descending size.
    clusters.sort(key=lambda c: len(c.elements), reverse=True)

    if _logger.isEnabledFor(logging.DEBUG):
        for idx, c in enumerate(clusters):
            _logger.debug(f"Cluster {idx} - Size: {len(c.elements)}")

    return clusters


def process_sequence(
    gt_poses: Sequence[np.ndarray], reloc_poses: Sequence[np.ndarray], chunk_size: int
) -> List[np.ndarray]:
    """Process a sequence of relocalisation results and robustify them using
    sequence-based information.

    The idea is that multiple relocalisation attempts from different frames
    (which are captured from different camera poses) should output consistent
    results. In real life though there will inevitable be some error/outliers/noisy
    results. We exploit this idea and, for each relocalised frame, we cluster
    the {chunk_size} previous relocalisation results. We then return a blended
    pose obtained from the largest cluster as "improved" relocalisation for
    that frame.

    :param gt_poses: A list camera poses from which the relocalisation was attempted.
        Expressed as 4x4 camera-to-world matrices, with the origin in the camera
        reference frame.
    :param reloc_poses: A list of relocalisation results. Expected as
        4x4 camera-to-world matrices, in the relocalisation reference frame.
        The origin of this does not have to coincide with the origin of the
        gt_poses (and probably never will, otherwise there would be no point in
        relocalising the camera).
    :param chunk_size: The number of previous frames to use to robustify
        each relocalisation result.
    :return: A List of refined relocalisation results.
    """
    refined_poses: List[np.ndarray] = []

    for frame_idx, (frame_w_T_c, frame_reloc_T_c) in enumerate(
        zip(gt_poses, reloc_poses)
    ):
        _logger.info(f"Processing frame {frame_idx}/{len(gt_poses)}.")
        _logger.debug(f"Camera pose: {frame_w_T_c}.")
        _logger.debug(f"Initial reloc pose: {frame_reloc_T_c}")

        # Extract chunk of gt and reloc poses before the current frame
        # INCLUDING the current frame
        start_idx = max(0, frame_idx + 1 - chunk_size)
        stop_idx = frame_idx + 1

        chunk_gt = gt_poses[start_idx:stop_idx]
        chunk_reloc = reloc_poses[start_idx:stop_idx]
        assert np.allclose(chunk_reloc[-1], frame_reloc_T_c, equal_nan=True)

        # Convert all the relocalisation poses in the last camera reference
        # frame.
        chunk_reloc_transformed = transform_sequence(chunk_reloc, chunk_gt, frame_w_T_c)
        # filter invalid relocalisations
        chunk_reloc_transformed = [
            x for x in chunk_reloc_transformed if np.isfinite(x).all()
        ]
        _logger.info(
            f"The chunk has {len(chunk_reloc)} relocalised poses, "
            f"of which {len(chunk_reloc_transformed)} are valid."
        )
        clustered_poses = cluster_poses(chunk_reloc_transformed)
        _logger.info(
            f"Clustered {len(chunk_reloc_transformed)} poses in "
            f"{len(clustered_poses)} clusters."
        )

        if clustered_poses:
            _logger.info(
                f"The largest cluster has {len(clustered_poses[0].elements)} poses."
            )
            _logger.debug(f"Centroid:\n{clustered_poses[0].centroid}")
            refined_pose = clustered_poses[0].centroid
        else:
            _logger.info(f"Cannot cluster poses. Using the original pose")
            refined_pose = frame_reloc_T_c

        # Output the pose.
        refined_poses.append(refined_pose)

    return refined_poses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute batched relocalisations.")
    parser.add_argument(
        "gt_file",
        type=Path,
        help="path to the file containing the ground truth poses (we assume "
        "frame to frame camera tracking is perfect)",
    )
    parser.add_argument(
        "reloc_file",
        type=Path,
        help="path to the file containing the relocalisation results",
    )
    parser.add_argument(
        "out_folder",
        type=Path,
        help="path to the folder that will contain the results",
    )
    parser.add_argument("--chunk-size", type=int, default=30, help="batch size")
    args = parser.parse_args()

    # Create the output folder.
    args.out_folder.mkdir(parents=True, exist_ok=True)

    gt_file = args.gt_file
    reloc_file = args.reloc_file
    seq_id = gt_file.stem

    # Load poses.
    gt_poses = load_sequence(gt_file)
    reloc_poses = load_sequence(reloc_file)
    assert len(gt_poses) == len(reloc_poses)

    # Process the sequence
    _logger.info(f"Processing seq: {seq_id} from {gt_file} and {reloc_file}")
    clustered_poses = process_sequence(gt_poses, reloc_poses, args.chunk_size)

    # Save the clustered poses.
    save_clustered_results(clustered_poses, args.out_folder, seq_id)
