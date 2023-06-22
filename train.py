# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import os
import sys
from typing import List
import io
import imageio
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_efficient_distloss import (
    eff_distloss,
    eff_distloss_native,
    flatten_eff_distloss,
)
from utils import visualize_depth_numpy


from camera import (
    pose_to_mtx,
    cam2world,
    lie,
    pose,
    procrustes_analysis,
    rotation_distance,
    get_novel_view_poses,
)

from dataLoader import dataset_dict
from dataLoader.ray_utils import (
    get_ray_directions_blender,
    get_ray_directions_lean,
    get_rays,
    get_rays_lean,
    get_rays_with_batch,
    ndc_rays_blender,
    ndc_rays_blender2,
)
from models.tensoRF import TensorVMSplit, TensorVMSplit_TimeEmbedding
from opt import config_parser
from renderer import (
    evaluation,
    evaluation_path,
    OctreeRender_trilinear_fast,
    render,
    induce_flow,
    render_3d_point,
    render_single_3d_point,
    NDC2world,
    induce_flow_single,
    raw2outputs,
    sampleXYZ,
    contract2world,
)
from utils import cal_n_samples, convert_sdf_samples_to_ply, N_to_reso, TVLoss
from flow_viz import flow_to_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


# Dummy tensorboard logger
class DummyWriter:
    def add_scalar(*args, **kwargs):
        pass

    def add_images(*args, **kwargs):
        pass


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr : self.curr + self.batch]


def ids2pixel(W, H, ids):
    """
    Regress pixel coordinates from
    """
    col = ids % W
    row = (ids // W) % H
    view_ids = ids // (W * H)
    return col, row, view_ids


@torch.no_grad()
def export_mesh(args):
    ckpt = None
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(
        alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005
    )


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


# from DynDyn
def generate_path(c2w, focal, sc, length=None):
    # hwf = c2w[:, 4:5]
    num_novelviews = 60
    max_disp = 48.0
    # H, W, focal = hwf[:, 0]
    # downsample = 2.0
    # focal = (854 / 2 * np.sqrt(3)) / float(downsample)

    max_trans = max_disp / focal[0] * sc
    dolly_poses = []
    dolly_focals = []

    # Dolly zoom
    for i in range(30):
        x_trans = 0.0
        y_trans = 0.0
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )
        i_pose = np.linalg.inv(i_pose)
        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )
        render_pose = np.dot(ref_pose, i_pose)
        dolly_poses.append(render_pose[:3, :])
        new_focal = focal[0] - focal[0] * 0.1 * z_trans / max_trans / 2.5
        dolly_focals.append(new_focal)
    dolly_poses = np.stack(dolly_poses, 0)[:, :3]

    zoom_poses = []
    zoom_focals = []
    # Zoom in
    for i in range(30):
        x_trans = 0.0
        y_trans = 0.0
        # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * args.z_trans_multiplier
        z_trans = max_trans * 2.5 * i / float(30 // 2)
        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        zoom_poses.append(render_pose[:3, :])
        zoom_focals.append(focal[0])
    zoom_poses = np.stack(zoom_poses, 0)[:, :3]

    spiral_poses = []
    spiral_focals = []
    # Rendering teaser. Add translation.
    for i in range(30):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans
            * 1.5
            * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0)
            * 2.0
            / 3.0
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        spiral_poses.append(render_pose[:3, :])
        spiral_focals.append(focal[0])
    spiral_poses = np.stack(spiral_poses, 0)[:, :3]

    fix_view_poses = []
    fix_view_focals = []
    # fix view
    for i in range(length):
        render_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        fix_view_poses.append(render_pose[:3, :])
        fix_view_focals.append(focal[0])
    fix_view_poses = np.stack(fix_view_poses, 0)[:, :3]

    change_view_time_poses = []
    change_view_time_focals = []
    # Rendering teaser. Add translation.
    for i in range(length):
        x_trans = max_trans * 1.5 * np.sin(2.0 * np.pi * float(i) / float(30)) * 2.0
        y_trans = (
            max_trans
            * 1.5
            * (np.cos(2.0 * np.pi * float(i) / float(30)) - 1.0)
            * 2.0
            / 3.0
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        change_view_time_poses.append(render_pose[:3, :])
        change_view_time_focals.append(focal[0])
    change_view_time_poses = np.stack(change_view_time_poses, 0)[:, :3]

    return (
        dolly_poses,
        dolly_focals,
        zoom_poses,
        zoom_focals,
        spiral_poses,
        spiral_focals,
        fix_view_poses,
        fix_view_focals,
        change_view_time_poses,
        change_view_time_focals,
    )


# from DynDyn
def generate_follow_spiral(c2ws, focal, sc):
    num_novelviews = int(c2ws.shape[0] * 2)
    max_disp = 48.0 * 2

    max_trans = max_disp / focal[0] * sc
    output_poses = []
    output_focals = []

    # Rendering teaser. Add translation.
    for i in range(c2ws.shape[0]):
        x_trans = (
            max_trans
            * np.sin(2.0 * np.pi * float(i) / float(num_novelviews) * 4.0)
            * 1.0
        )
        y_trans = (
            max_trans
            * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews) * 4.0) - 1.0)
            * 0.33
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [c2ws[i, :3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0
        )

        render_pose = np.dot(ref_pose, i_pose)
        # output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
        output_poses.append(render_pose[:3, :])
    # backward
    for i in range(c2ws.shape[0]):
        x_trans = (
            max_trans
            * np.sin(2.0 * np.pi * float(i) / float(num_novelviews) * 2.0)
            * 1.0
        )
        y_trans = (
            max_trans
            * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews) * 2.0) - 1.0)
            * 0.33
        )
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)

        ref_pose = np.concatenate(
            [
                c2ws[c2ws.shape[0] - 1 - i, :3, :4],
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(render_pose[:3, :])
    return output_poses


@torch.no_grad()
def render_test(args, logfolder):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        use_disp=args.use_disp,
        use_foreground_mask=args.use_foreground_mask,
    )
    white_bg = test_dataset.white_bg
    ray_type = args.ray_type

    if not os.path.exists(args.ckpt):
        raise RuntimeError("the ckpt path does not exists!!")

    # dynamic
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt["kwargs"]
    poses_mtx = kwargs.pop("se3_poses").to(device)
    focal_refine = kwargs.pop("focal_ratio_refine").to(device)
    kwargs.update({"device": device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    # static
    ckpt_static = torch.load(args.ckpt[:-3] + "_static.th", map_location=device)
    kwargs_static = ckpt_static["kwargs"]
    poses_mtx = kwargs_static.pop("se3_poses").to(device)
    focal_refine = kwargs_static.pop("focal_ratio_refine").to(device)
    kwargs_static.update({"device": device})
    tensorf_static = TensorVMSplit(**kwargs_static)
    tensorf_static.load(ckpt_static)
    os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
    np.save(f"{logfolder}/{args.expname}/poses.npy", poses_mtx.detach().cpu().numpy())
    np.save(
        f"{logfolder}/{args.expname}/focal.npy", focal_refine.detach().cpu().numpy()
    )

    if args.render_train:
        os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
        train_dataset = dataset(
            args.datadir, split="train", downsample=args.downsample_train, is_stack=True
        )
        PSNRs_test, _ = evaluation(
            train_dataset,
            poses_mtx,
            tensorf,
            args,
            renderer,
            f"{logfolder}/imgs_train_all",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
        )
        print(
            f"======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================"
        )

    if args.render_test:
        os.makedirs(f"{logfolder}/{args.expname}/imgs_test_all", exist_ok=True)
        _, near_fars, depth_test_all = evaluation(
            test_dataset,
            poses_mtx,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            args,
            renderer,
            f"{logfolder}/{args.expname}/imgs_test_all",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
        )

    if args.render_path:
        SE3_poses = poses_mtx

        mean_pose = torch.mean(poses_mtx[:, :, 3], 0)
        render_idx = 0
        best_dist = 1000000000
        for iidx in range(SE3_poses.shape[0]):
            cur_dist = torch.mean((SE3_poses[iidx, :, 3] - mean_pose) ** 2)
            if cur_dist < best_dist:
                best_dist = cur_dist
                render_idx = iidx
        print(render_idx)
        sc = near_fars[render_idx][0] * 0.75
        c2w = SE3_poses.cpu().detach().numpy()[render_idx]

        # Get average pose
        up_m = normalize(SE3_poses.cpu().detach().numpy()[:, :3, 1].sum(0))

        (
            dolly_poses,
            dolly_focals,
            zoom_poses,
            zoom_focals,
            spiral_poses,
            spiral_focals,
            fix_view_poses,
            fix_view_focals,
            change_view_time_poses,
            change_view_time_focals,
        ) = generate_path(
            SE3_poses.cpu().detach().numpy()[render_idx],
            focal=[focal_refine.item(), focal_refine.item()],
            sc=sc,
            length=SE3_poses.shape[0],
        )

        # fix view, change time
        os.makedirs(f"{logfolder}/{args.expname}/fix_view", exist_ok=True)
        _, depth_fix_view_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            fix_view_poses,
            renderer,
            f"{logfolder}/{args.expname}/fix_view",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=False,
            change_time="change",
            render_focal=fix_view_focals,
        )
        # change view, change time
        os.makedirs(f"{logfolder}/{args.expname}/change_view_time", exist_ok=True)
        _, depth_change_view_time_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            change_view_time_poses,
            renderer,
            f"{logfolder}/{args.expname}/change_view_time",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=False,
            change_time="change",
            render_focal=change_view_time_focals,
        )
        # dolly
        os.makedirs(f"{logfolder}/{args.expname}/dolly", exist_ok=True)
        _, depth_dolly_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            dolly_poses,
            renderer,
            f"{logfolder}/{args.expname}/dolly",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,
            render_focal=dolly_focals,
        )
        # zoom
        os.makedirs(f"{logfolder}/{args.expname}/zoom", exist_ok=True)
        _, depth_zoom_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            zoom_poses,
            renderer,
            f"{logfolder}/{args.expname}/zoom",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,
            render_focal=zoom_focals,
        )
        # spiral
        os.makedirs(f"{logfolder}/{args.expname}/spiral", exist_ok=True)
        _, depth_spiral_all = evaluation_path(
            test_dataset,
            focal_refine.cpu(),
            tensorf_static,
            tensorf,
            spiral_poses,
            renderer,
            f"{logfolder}/{args.expname}/spiral",
            N_vis=-1,
            N_samples=-1,
            white_bg=white_bg,
            ray_type=args.ray_type,
            device=device,
            change_view=True,
            change_time=render_idx / (args.N_voxel_t - 1) * 2.0 - 1.0,
            render_focal=spiral_focals,
        )

        all_depth = torch.stack(
            depth_test_all
            + depth_fix_view_all
            + depth_change_view_time_all
            + depth_dolly_all
            + depth_zoom_all
            + depth_spiral_all
        )
        depth_map_min = torch.quantile(all_depth[:, ::4, ::4], 0.05).item()
        depth_map_max = torch.quantile(all_depth[:, ::4, ::4], 0.95).item()

        for idx in range(len(depth_test_all)):
            depth_test_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_test_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/imgs_test_all/depthvideo.mp4",
            np.stack(depth_test_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_fix_view_all)):
            depth_fix_view_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_fix_view_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/fix_view/depthvideo.mp4",
            np.stack(depth_fix_view_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_change_view_time_all)):
            depth_change_view_time_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_change_view_time_all[idx],
                    min=depth_map_min,
                    max=depth_map_max,
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/change_view_time/depthvideo.mp4",
            np.stack(depth_change_view_time_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_dolly_all)):
            depth_dolly_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_dolly_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/dolly/depthvideo.mp4",
            np.stack(depth_dolly_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_zoom_all)):
            depth_zoom_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_zoom_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/zoom/depthvideo.mp4",
            np.stack(depth_zoom_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

        for idx in range(len(depth_spiral_all)):
            depth_spiral_all[idx] = visualize_depth_numpy(
                torch.clamp(
                    depth_spiral_all[idx], min=depth_map_min, max=depth_map_max
                ).numpy(),
                (depth_map_min, depth_map_max),
            )[0]
        imageio.mimwrite(
            f"{logfolder}/{args.expname}/spiral/depthvideo.mp4",
            np.stack(depth_spiral_all),
            fps=30,
            quality=8,
            format="ffmpeg",
            output_params=["-f", "mp4"],
        )

    return args.ckpt


@torch.no_grad()
def prealign_cameras(pose_in, pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3, device=pose_in.device)
    center_pred = cam2world(center, pose_in)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=pose_in.device))
    # align the camera poses
    center_aligned = (center_pred - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    R_aligned = pose_in[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    pose_aligned = pose(R=R_aligned, t=t_aligned)
    return pose_aligned, sim3


@torch.no_grad()
def evaluate_camera_alignment(pose_aligned, pose_GT):
    # measure errors in rotation and translation
    # pose_aligned: [N, 3, 4]
    # pose_GT:      [N, 3, 4]
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)  # [N, 3, 3], [N, 3, 1]
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
    return R_error, t_error


def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    # vertices = cam2world(vertices[None],pose)
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def compute_depth_loss(dyn_depth, gt_depth):
    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / (s_d + 1e-10)

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / (s_gt + 1e-10)

    # return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)
    return torch.sum((dyn_depth_norm - gt_depth_norm) ** 2)


def get_stats(X, norm=2):
    """
    :param X (N, H, W, C)
    :returns mean (1, 1, 1, C), scale (1)
    """
    mean = X.mean(dim=(0, 1, 2), keepdim=True)  # (1, 1, 1, C)
    if norm == 1:
        mag = torch.abs(X - mean).sum(dim=-1)  # (N, H, W)
    else:
        mag = np.sqrt(2) * torch.sqrt(torch.square(X - mean).sum(dim=-1))  # (N, H, W)
    scale = mag.mean() + 1e-6
    return mean, scale


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(
        args.datadir,
        split="train",
        downsample=args.downsample_train,
        is_stack=False,
        use_disp=args.use_disp,
        use_foreground_mask=args.use_foreground_mask,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
    )
    test_dataset = dataset(
        args.datadir,
        split="test",
        downsample=args.downsample_train,
        is_stack=True,
        use_disp=args.use_disp,
        use_foreground_mask=args.use_foreground_mask,
        with_GT_poses=args.with_GT_poses,
        ray_type=args.ray_type,
    )
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    W, H = train_dataset.img_wh

    # init resolution
    upsamp_list = args.upsamp_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f"{args.basedir}/{args.expname}"

    # init log fileinit log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    # static TensoRF
    tensorf_static = TensorVMSplit(
        aabb,
        reso_cur,
        args.N_voxel_t,
        device,
        density_n_comp=n_lamb_sigma,
        appearance_n_comp=n_lamb_sh,
        app_dim=args.data_dim_color,
        near_far=near_far,
        shadingMode=args.shadingModeStatic,
        alphaMask_thres=args.alpha_mask_thre,
        density_shift=args.density_shift,
        distance_scale=args.distance_scale,
        pos_pe=args.pos_pe,
        view_pe=args.view_pe,
        fea_pe=2,
        featureC=args.featureC,
        step_ratio=args.step_ratio,
        fea2denseAct=args.fea2denseAct,
    )

    # dynamic tensorf
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb,
            reso_cur,
            args.N_voxel_t,
            device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=args.data_dim_color,
            near_far=near_far,
            shadingMode=args.shadingMode,
            alphaMask_thres=args.alpha_mask_thre,
            density_shift=args.density_shift,
            distance_scale=args.distance_scale,
            pos_pe=args.pos_pe,
            view_pe=args.view_pe,
            fea_pe=0,
            featureC=args.featureC,
            step_ratio=args.step_ratio,
            fea2denseAct=args.fea2denseAct,
        )

    grad_vars = tensorf_static.get_optparam_groups(args.lr_init, args.lr_basis)
    grad_vars.extend(tensorf.get_optparam_groups(args.lr_init, args.lr_basis))
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (
        torch.round(
            torch.exp(
                torch.linspace(
                    np.log(args.N_voxel_init),
                    np.log(args.N_voxel_final),
                    len(upsamp_list) + 1,
                )
            )
        ).long()
    ).tolist()[1:]

    PSNRs, PSNRs_test = [], [0]

    allrgbs = train_dataset.all_rgbs
    allts = train_dataset.all_ts
    if args.with_GT_poses:
        allposes = train_dataset.all_poses  # (12, 3, 4)
    allflows_f = train_dataset.all_flows_f.to(device)
    allflowmasks_f = train_dataset.all_flow_masks_f.to(device)
    allflows_b = train_dataset.all_flows_b.to(device)
    allflowmasks_b = train_dataset.all_flow_masks_b.to(device)

    if args.use_disp:
        alldisps = train_dataset.all_disps
    allforegroundmasks = train_dataset.all_foreground_masks

    init_poses = torch.zeros(args.N_voxel_t, 9)
    if args.with_GT_poses:
        init_poses[..., 0:3] = allposes[..., :, 0]
        init_poses[..., 3:6] = allposes[..., :, 1]
        init_poses[..., 6:9] = allposes[..., :, 3]
    else:
        init_poses[..., 0] = 1
        init_poses[..., 4] = 1
    poses_refine = torch.nn.Embedding(args.N_voxel_t, 9).to(device)
    poses_refine.weight.data.copy_(init_poses.to(device))

    # optimizing focal length
    fov_refine_embedding = torch.nn.Embedding(1, 1).to(device)
    fov_refine_embedding.weight.data.copy_(
        torch.ones(1, 1).to(device) * 30 / 180 * np.pi
    )
    if args.with_GT_poses:
        focal_refine = torch.tensor(train_dataset.focal[0]).to(device)

    ii, jj = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    grid = torch.from_numpy(np.stack([ii, jj], -1)).to(device)
    grid = torch.tile(torch.unsqueeze(grid, 0), (args.N_voxel_t, 1, 1, 1))
    allgrids = grid.view(-1, 2)

    # setup optimizer
    if args.optimize_poses:
        lr_pose = 3e-3
        lr_pose_end = 1e-5  # 5:X, 10:X
        optimizer_pose = torch.optim.Adam(poses_refine.parameters(), lr=lr_pose)
        gamma = (lr_pose_end / lr_pose) ** (
            1.0 / (args.n_iters // 2 - args.upsamp_list[-1])
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_pose, gamma=gamma)

    if args.optimize_focal_length:
        lr_pose = 3e-3
        lr_pose_end = 1e-5  # 5:X, 10:X
        optimizer_focal = torch.optim.Adam(fov_refine_embedding.parameters(), lr=0.0)
        gamma = (lr_pose_end / lr_pose) ** (
            1.0 / (args.n_iters // 2 - args.upsamp_list[-1])
        )
        scheduler_focal = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_focal, gamma=gamma
        )

    trainingSampler = SimpleSampler(allts.shape[0], args.batch_size)
    trainingSampler_2 = SimpleSampler(allts.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    distortion_weight_static, distortion_weight_dynamic = (
        args.distortion_weight_static,
        args.distortion_weight_dynamic,
    )
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    decay_iteration = 100

    pbar = tqdm(
        range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout
    )
    for iteration in pbar:
        # Lambda decay.
        Temp_static = 1.0 / (10 ** (iteration / (100000)))
        Temp = 1.0 / (10 ** (iteration // (decay_iteration * 1000)))
        Temp_disp_TV = 1.0 / (10 ** (iteration // (50000)))

        if args.optimize_focal_length:
            focal_refine = (
                np.maximum(H, W) / 2.0 / torch.tan(fov_refine_embedding.weight[0, 0])
            )

        ray_idx = trainingSampler.nextids()

        rgb_train, ts_train, grid_train = (
            allrgbs[ray_idx].to(device),
            allts[ray_idx].to(device),
            allgrids[ray_idx],
        )
        flow_f_train, flow_mask_f_train, flow_b_train, flow_mask_b_train = (
            allflows_f[ray_idx],
            allflowmasks_f[ray_idx][..., None],
            allflows_b[ray_idx],
            allflowmasks_b[ray_idx][..., None],
        )

        if args.use_disp:
            alldisps_train = alldisps[ray_idx].to(device)

        allforegroundmasks_train = allforegroundmasks[ray_idx].to(device)

        poses_refine2 = poses_refine.weight.clone()
        poses_refine2[..., 6:9] = poses_refine2[..., 6:9]
        poses_mtx = pose_to_mtx(poses_refine2)

        i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))

        directions = get_ray_directions_lean(
            i, j, [focal_refine, focal_refine], [W / 2, H / 2]
        )
        poses_mtx_batched = poses_mtx[view_ids]
        rays_o, rays_d = get_rays_lean(directions, poses_mtx_batched)  # both (b, 3)
        if args.ray_type == "ndc":
            rays_o, rays_d = ndc_rays_blender2(
                H, W, [focal_refine, focal_refine], 1.0, rays_o, rays_d
            )
        rays_train = torch.cat([rays_o, rays_d], -1).view(-1, 6)

        t_ref = ray_idx // (H * W)
        u_ref = (ray_idx % (H * W)) // W  # height
        v_ref = (ray_idx % (H * W)) % W  # width
        t_interval = 2 / (args.N_voxel_t - 1)

        # index the pose for forward and backward
        allposes_refine_f = torch.cat((poses_mtx[1:], poses_mtx[-1:]), 0)
        allposes_refine_b = torch.cat((poses_mtx[0:1], poses_mtx[:-1]), 0)
        allposes_refine_f_train = allposes_refine_f[t_ref]
        allposes_refine_b_train = allposes_refine_b[t_ref]

        total_loss = 0.0

        xyz_sampled, z_vals, ray_valid = sampleXYZ(
            tensorf,
            rays_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )
        # static tensorf
        _, _, _, _, _, _, rgb_points_static, sigmas_static, _, _ = tensorf_static(
            rays_train.detach(),
            ts_train,
            None,
            xyz_sampled,
            z_vals,
            ray_valid,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        # dynamic tensorf
        (
            _,
            _,
            blending,
            pts_ref,
            _,
            _,
            rgb_points_dynamic,
            sigmas_dynamic,
            z_vals_dynamic,
            dists_dynamic,
        ) = tensorf(
            rays_train.detach(),
            ts_train,
            None,
            xyz_sampled,
            z_vals,
            ray_valid,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )

        (
            rgb_map_full,
            _,
            _,
            _,
            rgb_map_s,
            depth_map_s,
            _,
            weights_s,
            rgb_map_d,
            depth_map_d,
            _,
            weights_d,
            dynamicness_map,
        ) = raw2outputs(
            rgb_points_static.detach(),
            sigmas_static.detach(),
            rgb_points_dynamic,
            sigmas_dynamic,
            dists_dynamic,
            blending,
            z_vals_dynamic,
            rays_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )

        # novel mask zero loss
        # sample training view and novel time combination
        ray_idx_rand = trainingSampler_2.nextids()
        ts_train_rand = allts[ray_idx_rand].to(device)
        xyz_sampled_rand, z_vals_rand, ray_valid_rand = sampleXYZ(
            tensorf,
            rays_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )
        (
            _,
            _,
            _,
            _,
            _,
            _,
            rgb_points_static_rand,
            sigmas_static_rand,
            _,
            _,
        ) = tensorf_static(
            rays_train.detach(),
            ts_train_rand,
            None,
            xyz_sampled_rand,
            z_vals_rand,
            ray_valid_rand,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            blending_rand,
            _,
            _,
            _,
            rgb_points_dynamic_rand,
            sigmas_dynamic_rand,
            z_vals_dynamic_rand,
            dists_dynamic_rand,
        ) = tensorf(
            rays_train.detach(),
            ts_train_rand,
            None,
            xyz_sampled_rand,
            z_vals_rand,
            ray_valid_rand,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            _,
            _,
            _,
            depth_map_s_rand,
            _,
            _,
            _,
            depth_map_d_rand,
            _,
            weights_d_rand,
            dynamicness_map_rand,
        ) = raw2outputs(
            rgb_points_static_rand.detach(),
            sigmas_static_rand.detach(),
            rgb_points_dynamic_rand,
            sigmas_dynamic_rand,
            dists_dynamic_rand,
            blending_rand,
            z_vals_dynamic_rand,
            rays_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )

        if iteration >= args.upsamp_list[3]:
            # skewed mask loss
            clamped_mask_rand = torch.clamp(
                dynamicness_map_rand, min=1e-6, max=1.0 - 1e-6
            )
            skewed_mask_loss_rand = torch.mean(
                -(
                    (clamped_mask_rand**2) * torch.log((clamped_mask_rand**2))
                    + (1 - (clamped_mask_rand**2))
                    * torch.log(1 - (clamped_mask_rand**2))
                )
            )
            total_loss += 0.01 * skewed_mask_loss_rand
            summary_writer.add_scalar(
                "train/skewed_mask_loss_rand",
                skewed_mask_loss_rand.detach().item(),
                global_step=iteration,
            )

            novel_view_time_mask_loss = torch.mean(torch.abs(dynamicness_map_rand))
            total_loss += 0.01 * novel_view_time_mask_loss
            summary_writer.add_scalar(
                "train/novel_view_time_mask_loss",
                novel_view_time_mask_loss.detach().item(),
                global_step=iteration,
            )

        # novel adaptive Order loss
        if args.ray_type == "ndc":
            novel_order_loss = torch.sum(
                ((depth_map_d_rand - depth_map_s_rand.detach()) ** 2)
                * (1.0 - dynamicness_map_rand.detach())
            ) / (torch.sum(1.0 - dynamicness_map_rand.detach()) + 1e-8)
        elif args.ray_type == "contract":
            novel_order_loss = torch.sum(
                (
                    (
                        1.0 / (depth_map_d_rand + 1e-6)
                        - 1.0 / (depth_map_s_rand.detach() + 1e-6)
                    )
                    ** 2
                )
                * (1.0 - dynamicness_map_rand.detach())
            ) / (torch.sum((1.0 - dynamicness_map_rand.detach())) + 1e-8)
        total_loss += novel_order_loss * 10.0
        summary_writer.add_scalar(
            "train/novel_order_loss",
            (novel_order_loss).detach().item(),
            global_step=iteration,
        )

        if distortion_weight_dynamic > 0:
            ray_id = torch.tile(
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],
                (1, weights_d_rand.shape[1]),
            ).to(device)
            loss_distortion = flatten_eff_distloss(
                torch.flatten(weights_d_rand),
                torch.flatten(z_vals_dynamic_rand.detach()),
                1 / (weights_d_rand.shape[1]),
                torch.flatten(ray_id),
            )
            total_loss += (
                loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            )
            summary_writer.add_scalar(
                "train/loss_distortion_rand",
                (loss_distortion).detach().item(),
                global_step=iteration,
            )

        scene_flow_f, scene_flow_b = tensorf.get_forward_backward_scene_flow(
            pts_ref, ts_train.to(device)
        )

        loss = torch.mean((rgb_map_full - rgb_train) ** 2)
        PSNRs.append(-10.0 * np.log(loss.detach().item()) / np.log(10.0))
        summary_writer.add_scalar("train/PSNR", PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar(
            "train/mse", loss.detach().item(), global_step=iteration
        )
        total_loss += 3.0 * loss

        img_d_loss = torch.mean((rgb_map_d - rgb_train) ** 2)
        total_loss += 1.0 * img_d_loss
        summary_writer.add_scalar(
            "train/img_d_loss", img_d_loss.detach().item(), global_step=iteration
        )

        # Flow grouping loss

        if iteration >= args.upsamp_list[0]:
            # mask loss
            mask_loss = torch.mean(
                torch.abs(dynamicness_map - allforegroundmasks_train[..., 0])
            )
            total_loss += 0.1 * mask_loss * Temp_disp_TV
            summary_writer.add_scalar(
                "train/mask_loss", mask_loss.detach().item(), global_step=iteration
            )

        if iteration >= args.upsamp_list[3]:
            # skewed mask loss
            clamped_mask = torch.clamp(dynamicness_map, min=1e-6, max=1.0 - 1e-6)
            skewed_mask_loss = torch.mean(
                -(
                    (clamped_mask**2) * torch.log((clamped_mask**2))
                    + (1 - (clamped_mask**2)) * torch.log(1 - (clamped_mask**2))
                )
            )
            total_loss += 0.01 * skewed_mask_loss
            summary_writer.add_scalar(
                "train/skewed_mask_loss",
                skewed_mask_loss.detach().item(),
                global_step=iteration,
            )

            mask_L1_reg_loss = torch.mean(torch.abs(dynamicness_map))
            total_loss += 0.01 * mask_L1_reg_loss
            summary_writer.add_scalar(
                "train/mask_L1_reg_loss",
                mask_L1_reg_loss.detach().item(),
                global_step=iteration,
            )

        if args.ray_type == "ndc":
            pts_f = pts_ref + scene_flow_f
            pts_b = pts_ref + scene_flow_b
        elif args.ray_type == "contract":
            pts_f = torch.clamp(pts_ref + scene_flow_f, min=-2.0 + 1e-6, max=2.0 - 1e-6)
            pts_b = torch.clamp(pts_ref + scene_flow_b, min=-2.0 + 1e-6, max=2.0 - 1e-6)

        induced_flow_f, induced_disp_f = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),
            weights_d,
            pts_f,
            grid_train,
            rays_train.detach(),
            ray_type=args.ray_type,
        )
        flow_f_loss = (
            torch.sum(torch.abs(induced_flow_f - flow_f_train) * flow_mask_f_train)
            / (torch.sum(flow_mask_f_train) + 1e-8)
            / flow_f_train.shape[-1]
        )
        total_loss += 0.02 * flow_f_loss * Temp
        induced_flow_b, induced_disp_b = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d,
            pts_b,
            grid_train,
            rays_train.detach(),
            ray_type=args.ray_type,
        )
        flow_b_loss = (
            torch.sum(torch.abs(induced_flow_b - flow_b_train) * flow_mask_b_train)
            / (torch.sum(flow_mask_b_train) + 1e-8)
            / flow_b_train.shape[-1]
        )
        total_loss += 0.02 * flow_b_loss * Temp
        summary_writer.add_scalar(
            "train/flow_f_loss", flow_f_loss.detach().item(), global_step=iteration
        )
        summary_writer.add_scalar(
            "train/flow_b_loss", flow_b_loss.detach().item(), global_step=iteration
        )

        small_scene_flow_loss = torch.mean(torch.abs(scene_flow_f)) + torch.mean(
            torch.abs(scene_flow_b)
        )
        total_loss += args.small_scene_flow_weight * small_scene_flow_loss
        summary_writer.add_scalar(
            "train/small_scene_flow_loss",
            small_scene_flow_loss.detach().item(),
            global_step=iteration,
        )

        # disparity loss
        # forward
        uv_f = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)
            + flow_f_train
        )
        directions_f = torch.stack(
            [
                (uv_f[..., 0] - W / 2) / (focal_refine.detach()),
                -(uv_f[..., 1] - H / 2) / (focal_refine.detach()),
                -torch.ones_like(uv_f[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_f_o, rays_f_d = get_rays_lean(directions_f, allposes_refine_f_train)
        if args.ray_type == "ndc":
            rays_f_o, rays_f_d = ndc_rays_blender2(
                H,
                W,
                [focal_refine.detach(), focal_refine.detach()],
                1.0,
                rays_f_o,
                rays_f_d,
            )
        rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)
        xyz_sampled_f, z_vals_f, ray_valid_f = sampleXYZ(
            tensorf,
            rays_f_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )

        _, _, _, _, _, _, rgb_points_static_f, sigmas_static_f, _, _ = tensorf_static(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            xyz_sampled_f,
            z_vals_f,
            ray_valid_f.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            blending_f,
            pts_ref_ff,
            _,
            _,
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            z_vals_dynamic_f,
            dists_dynamic_f,
        ) = tensorf(
            rays_f_train.detach(),
            ts_train + t_interval,
            None,
            xyz_sampled_f,
            z_vals_f,
            ray_valid_f.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        _, _, _, _, _, _, _, _, _, _, _, weights_d_f, _ = raw2outputs(
            rgb_points_static_f.detach(),
            sigmas_static_f.detach(),
            rgb_points_dynamic_f,
            sigmas_dynamic_f,
            dists_dynamic_f,
            blending_f,
            z_vals_dynamic_f,
            rays_f_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )
        _, induced_disp_ff = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_f_train.detach(),
            weights_d_f,
            pts_ref_ff,
            grid_train,
            rays_f_train.detach(),
            ray_type=args.ray_type,
        )
        disp_f_loss = torch.sum(
            torch.abs(induced_disp_f - induced_disp_ff) * flow_mask_f_train
        ) / (torch.sum(flow_mask_f_train) + 1e-8)
        total_loss += 0.04 * disp_f_loss * Temp
        summary_writer.add_scalar(
            "train/disp_f_loss", disp_f_loss.detach().item(), global_step=iteration
        )
        # backward
        uv_b = (
            torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)
            + flow_b_train
        )
        directions_b = torch.stack(
            [
                (uv_b[..., 0] - W / 2) / (focal_refine.detach()),
                -(uv_b[..., 1] - H / 2) / (focal_refine.detach()),
                -torch.ones_like(uv_b[..., 0]),
            ],
            -1,
        )  # (H, W, 3)
        rays_b_o, rays_b_d = get_rays_lean(directions_b, allposes_refine_b_train)
        if args.ray_type == "ndc":
            rays_b_o, rays_b_d = ndc_rays_blender2(
                H,
                W,
                [focal_refine.detach(), focal_refine.detach()],
                1.0,
                rays_b_o,
                rays_b_d,
            )
        rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)
        xyz_sampled_b, z_vals_b, ray_valid_b = sampleXYZ(
            tensorf,
            rays_b_train.detach(),
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )

        _, _, _, _, _, _, rgb_points_static_b, sigmas_static_b, _, _ = tensorf_static(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            xyz_sampled_b,
            z_vals_b,
            ray_valid_b.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        (
            _,
            _,
            blending_b,
            pts_ref_bb,
            _,
            _,
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            z_vals_dynamic_b,
            dists_dynamic_b,
        ) = tensorf(
            rays_b_train.detach(),
            ts_train - t_interval,
            None,
            xyz_sampled_b,
            z_vals_b,
            ray_valid_b.detach(),
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        _, _, _, _, _, _, _, _, _, _, _, weights_d_b, _ = raw2outputs(
            rgb_points_static_b.detach(),
            sigmas_static_b.detach(),
            rgb_points_dynamic_b,
            sigmas_dynamic_b,
            dists_dynamic_b,
            blending_b,
            z_vals_dynamic_b,
            rays_b_train.detach(),
            is_train=True,
            ray_type=args.ray_type,
        )
        _, induced_disp_bb = induce_flow(
            H,
            W,
            focal_refine.detach(),
            allposes_refine_b_train.detach(),
            weights_d_b,
            pts_ref_bb,
            grid_train,
            rays_b_train.detach(),
            ray_type=args.ray_type,
        )
        disp_b_loss = torch.sum(
            torch.abs(induced_disp_b - induced_disp_bb) * flow_mask_b_train
        ) / (torch.sum(flow_mask_b_train) + 1e-8)
        total_loss += 0.04 * disp_b_loss * Temp
        summary_writer.add_scalar(
            "train/disp_b_loss", disp_b_loss.detach().item(), global_step=iteration
        )

        smooth_scene_flow_loss = torch.mean(torch.abs(scene_flow_f + scene_flow_b))
        total_loss += smooth_scene_flow_loss * args.smooth_scene_flow_weight
        summary_writer.add_scalar(
            "train/smooth_scene_flow_loss",
            smooth_scene_flow_loss.detach().item(),
            global_step=iteration,
        )

        # Monocular depth loss
        total_mono_depth_loss = 0.0
        counter = 0.0
        total_mono_depth_loss_list = []
        counter_list = []
        for cam_idx in range(args.N_voxel_t):
            valid = t_ref == cam_idx
            if torch.sum(valid) > 1.0:
                if args.ray_type == "ndc":
                    total_mono_depth_loss += compute_depth_loss(
                        depth_map_d[valid], -alldisps_train[valid]
                    )
                elif args.ray_type == "contract":
                    total_mono_depth_loss += compute_depth_loss(
                        1.0 / (depth_map_d[valid] + 1e-6), alldisps_train[valid]
                    )
                    total_mono_depth_loss_list.append(
                        compute_depth_loss(
                            1.0 / (depth_map_d[valid] + 1e-6), alldisps_train[valid]
                        )
                    )
                counter += torch.sum(valid)
                counter_list.append(valid)
        total_mono_depth_loss = total_mono_depth_loss / counter
        total_loss += total_mono_depth_loss * args.monodepth_weight_dynamic * Temp
        summary_writer.add_scalar(
            "train/total_mono_depth_loss_dynamic",
            total_mono_depth_loss.detach().item(),
            global_step=iteration,
        )

        # adaptive Order loss
        if args.ray_type == "ndc":
            order_loss = torch.sum(
                ((depth_map_d - depth_map_s.detach()) ** 2)
                * (1.0 - dynamicness_map.detach())
            ) / (torch.sum(1.0 - dynamicness_map.detach()) + 1e-8)
        elif args.ray_type == "contract":
            order_loss = torch.sum(
                (
                    (1.0 / (depth_map_d + 1e-6) - 1.0 / (depth_map_s.detach() + 1e-6))
                    ** 2
                )
                * (1.0 - dynamicness_map.detach())
            ) / (torch.sum((1.0 - dynamicness_map.detach())) + 1e-8)
        total_loss += order_loss * 10.0
        summary_writer.add_scalar(
            "train/order_loss", (order_loss).detach().item(), global_step=iteration
        )

        # distortion loss from DVGO
        if distortion_weight_dynamic > 0:
            ray_id = torch.tile(
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],
                (1, weights_d.shape[1]),
            ).to(device)
            loss_distortion = flatten_eff_distloss(
                torch.flatten(weights_d),
                torch.flatten(z_vals_dynamic.detach()),
                1 / (weights_d.shape[1]),
                torch.flatten(ray_id),
            )
            loss_distortion += flatten_eff_distloss(
                torch.flatten(weights_d_f),
                torch.flatten(z_vals_dynamic_f.detach()),
                1 / (weights_d_f.shape[1]),
                torch.flatten(ray_id),
            )
            loss_distortion += flatten_eff_distloss(
                torch.flatten(weights_d_b),
                torch.flatten(z_vals_dynamic_b.detach()),
                1 / (weights_d_b.shape[1]),
                torch.flatten(ray_id),
            )
            total_loss += (
                loss_distortion * distortion_weight_dynamic * (iteration / args.n_iters)
            )
            summary_writer.add_scalar(
                "train/loss_distortion",
                (loss_distortion).detach().item(),
                global_step=iteration,
            )

        # TV losses
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar(
                "train/reg", loss_reg.detach().item(), global_step=iteration
            )
        if L1_reg_weight > 0:
            loss_reg_L1_density = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1_density
            summary_writer.add_scalar(
                "train/loss_reg_L1_density",
                loss_reg_L1_density.detach().item(),
                global_step=iteration,
            )

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_density", loss_tv.detach().item(), global_step=iteration
            )
            # TV for blending
            loss_tv = tensorf.TV_loss_blending(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_blending", loss_tv.detach().item(), global_step=iteration
            )
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar(
                "train/reg_tv_app", loss_tv.detach().item(), global_step=iteration
            )

        # static part for pose estimation
        xyz_sampled, z_vals, ray_valid = sampleXYZ(
            tensorf,
            rays_train,
            N_samples=nSamples,
            ray_type=args.ray_type,
            is_train=True,
        )
        # static tensorf
        (
            _,
            _,
            _,
            pts_ref_s,
            _,
            _,
            rgb_points_static,
            sigmas_static,
            _,
            _,
        ) = tensorf_static(
            rays_train,
            ts_train,
            None,
            xyz_sampled,
            z_vals,
            ray_valid,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )
        # dynamic tensorf
        (
            _,
            _,
            blending,
            pts_ref,
            _,
            _,
            rgb_points_dynamic,
            sigmas_dynamic,
            z_vals_dynamic,
            dists_dynamic,
        ) = tensorf(
            rays_train,
            ts_train,
            None,
            xyz_sampled,
            z_vals,
            ray_valid,
            is_train=True,
            white_bg=white_bg,
            ray_type=args.ray_type,
            N_samples=nSamples,
        )

        _, _, _, _, rgb_map_s, depth_map_s, _, weights_s, _, _, _, _, _ = raw2outputs(
            rgb_points_static,
            sigmas_static,
            rgb_points_dynamic,
            sigmas_dynamic,
            dists_dynamic,
            blending,
            z_vals_dynamic,
            rays_train,
            is_train=True,
            ray_type=args.ray_type,
        )

        ### static losses
        # RGB loss
        img_s_loss = (
            torch.sum(
                (rgb_map_s - rgb_train) ** 2
                * (1.0 - allforegroundmasks_train[..., 0:1])
            )
            / (torch.sum((1.0 - allforegroundmasks_train[..., 0:1])) + 1e-8)
            / rgb_map_s.shape[-1]
        )
        total_loss += 1.0 * img_s_loss
        summary_writer.add_scalar(
            "train/img_s_loss", img_s_loss.detach().item(), global_step=iteration
        )

        # static distortion loss from DVGO
        if distortion_weight_static > 0:
            ray_id = torch.tile(
                torch.range(0, args.batch_size - 1, dtype=torch.int64)[:, None],
                (1, weights_s.shape[1]),
            ).to(device)
            loss_distortion_static = flatten_eff_distloss(
                torch.flatten(weights_s),
                torch.flatten(z_vals),
                1 / (weights_s.shape[1]),
                torch.flatten(ray_id),
            )
            total_loss += (
                loss_distortion_static
                * distortion_weight_static
                * (iteration / args.n_iters)
            )
            summary_writer.add_scalar(
                "train/loss_distortion_static",
                (loss_distortion_static).detach().item(),
                global_step=iteration,
            )

        if L1_reg_weight > 0:
            loss_reg_L1_density_s = tensorf_static.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1_density_s
            summary_writer.add_scalar(
                "train/loss_reg_L1_density_s",
                loss_reg_L1_density_s.detach().item(),
                global_step=iteration,
            )

        if TV_weight_density > 0:
            loss_tv_static = tensorf_static.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar(
                "train/reg_tv_density_static",
                loss_tv_static.detach().item(),
                global_step=iteration,
            )
        if TV_weight_app > 0:
            loss_tv_static = tensorf_static.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv_static
            summary_writer.add_scalar(
                "train/reg_tv_app_static",
                loss_tv_static.detach().item(),
                global_step=iteration,
            )

        summary_writer.add_scalar(
            "train/focal_ratio_refine",
            focal_refine.detach().item(),
            global_step=iteration,
        )

        if args.optimize_poses:
            # static motion loss
            induced_flow_f_s, induced_disp_f_s = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_f_train,
                weights_s,
                pts_ref_s,
                grid_train,
                rays_train,
                ray_type=args.ray_type,
            )
            flow_f_s_loss = (
                torch.sum(
                    torch.abs(induced_flow_f_s - flow_f_train)
                    * flow_mask_f_train
                    * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                / (
                    torch.sum(
                        flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
                    )
                    + 1e-8
                )
                / flow_f_train.shape[-1]
            )
            total_loss += 0.02 * flow_f_s_loss * Temp_static
            induced_flow_b_s, induced_disp_b_s = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_b_train,
                weights_s,
                pts_ref_s,
                grid_train,
                rays_train,
                ray_type=args.ray_type,
            )
            flow_b_s_loss = (
                torch.sum(
                    torch.abs(induced_flow_b_s - flow_b_train)
                    * flow_mask_b_train
                    * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                / (
                    torch.sum(
                        flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
                    )
                    + 1e-8
                )
                / flow_b_train.shape[-1]
            )
            total_loss += 0.02 * flow_b_s_loss * Temp_static
            summary_writer.add_scalar(
                "train/flow_f_s_loss",
                flow_f_s_loss.detach().item(),
                global_step=iteration,
            )
            summary_writer.add_scalar(
                "train/flow_b_s_loss",
                flow_b_s_loss.detach().item(),
                global_step=iteration,
            )

            # static disparity loss
            # forward
            uv_f = (
                torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_f_train.device)
                + flow_f_train
            )
            directions_f = torch.stack(
                [
                    (uv_f[..., 0] - W / 2) / (focal_refine),
                    -(uv_f[..., 1] - H / 2) / (focal_refine),
                    -torch.ones_like(uv_f[..., 0]),
                ],
                -1,
            )  # (H, W, 3)
            rays_f_o, rays_f_d = get_rays_lean(
                directions_f, allposes_refine_f_train
            )  # both (b, 3)
            if args.ray_type == "ndc":
                rays_f_o, rays_f_d = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_f_o, rays_f_d
                )
            rays_f_train = torch.cat([rays_f_o, rays_f_d], -1).view(-1, 6)
            xyz_sampled_f, z_vals_f, ray_valid_f = sampleXYZ(
                tensorf_static,
                rays_f_train,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            _, _, _, pts_ref_s_ff, weights_s_ff, _, _, _, _, _ = tensorf_static(
                rays_f_train,
                ts_train,
                None,
                xyz_sampled_f,
                z_vals_f,
                ray_valid_f,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            _, induced_disp_s_ff = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_f_train,
                weights_s_ff,
                pts_ref_s_ff,
                grid_train,
                rays_f_train,
                ray_type=args.ray_type,
            )
            disp_f_s_loss = torch.sum(
                torch.abs(induced_disp_f_s - induced_disp_s_ff)
                * flow_mask_f_train
                * (1.0 - allforegroundmasks_train[..., 0:1])
            ) / (
                torch.sum(
                    flow_mask_f_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                + 1e-8
            )
            total_loss += 0.04 * disp_f_s_loss * Temp_static
            summary_writer.add_scalar(
                "train/disp_f_s_loss",
                disp_f_s_loss.detach().item(),
                global_step=iteration,
            )
            # backward
            uv_b = (
                torch.stack((v_ref + 0.5, u_ref + 0.5), -1).to(flow_b_train.device)
                + flow_b_train
            )
            directions_b = torch.stack(
                [
                    (uv_b[..., 0] - W / 2) / (focal_refine),
                    -(uv_b[..., 1] - H / 2) / (focal_refine),
                    -torch.ones_like(uv_b[..., 0]),
                ],
                -1,
            )  # (H, W, 3)
            rays_b_o, rays_b_d = get_rays_lean(
                directions_b, allposes_refine_b_train
            )  # both (b, 3)
            if args.ray_type == "ndc":
                rays_b_o, rays_b_d = ndc_rays_blender2(
                    H, W, [focal_refine, focal_refine], 1.0, rays_b_o, rays_b_d
                )
            rays_b_train = torch.cat([rays_b_o, rays_b_d], -1).view(-1, 6)
            xyz_sampled_b, z_vals_b, ray_valid_b = sampleXYZ(
                tensorf_static,
                rays_b_train,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            _, _, _, pts_ref_s_bb, weights_s_bb, _, _, _, _, _ = tensorf_static(
                rays_b_train,
                ts_train,
                None,
                xyz_sampled_b,
                z_vals_b,
                ray_valid_b,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            _, induced_disp_s_bb = induce_flow(
                H,
                W,
                focal_refine,
                allposes_refine_b_train,
                weights_s_bb,
                pts_ref_s_bb,
                grid_train,
                rays_b_train,
                ray_type=args.ray_type,
            )
            disp_b_s_loss = torch.sum(
                torch.abs(induced_disp_b_s - induced_disp_s_bb)
                * flow_mask_b_train
                * (1.0 - allforegroundmasks_train[..., 0:1])
            ) / (
                torch.sum(
                    flow_mask_b_train * (1.0 - allforegroundmasks_train[..., 0:1])
                )
                + 1e-8
            )
            total_loss += 0.04 * disp_b_s_loss * Temp_static
            summary_writer.add_scalar(
                "train/disp_b_s_loss",
                disp_b_s_loss.detach().item(),
                global_step=iteration,
            )

            # Monocular depth loss with mask for static TensoRF
            total_mono_depth_loss = 0.0
            counter = 0.0
            for cam_idx in range(args.N_voxel_t):
                valid = torch.bitwise_and(
                    t_ref == cam_idx, allforegroundmasks_train[..., 0].cpu() < 0.5
                )
                if torch.sum(valid) > 1.0:
                    if args.ray_type == "ndc":
                        total_mono_depth_loss += compute_depth_loss(
                            depth_map_s[valid], -alldisps_train[valid]
                        )
                    elif args.ray_type == "contract":
                        total_mono_depth_loss += compute_depth_loss(
                            1.0 / (depth_map_s[valid] + 1e-6), alldisps_train[valid]
                        )
                    counter += torch.sum(valid)
            total_mono_depth_loss = total_mono_depth_loss / counter
            total_loss += (
                total_mono_depth_loss * args.monodepth_weight_static * Temp_static
            )
            summary_writer.add_scalar(
                "train/total_mono_depth_loss_static",
                total_mono_depth_loss.detach().item(),
                global_step=iteration,
            )

            # sample for patch TV loss
            i, j, view_ids = ids2pixel(W, H, ray_idx.to(device))
            i_neighbor = torch.clamp(i + 1, max=W - 1)
            j_neighbor = torch.clamp(j + 1, max=H - 1)
            directions_i_neighbor = get_ray_directions_lean(
                i_neighbor, j, [focal_refine, focal_refine], [W / 2, H / 2]
            )
            rays_o_i_neighbor, rays_d_i_neighbor = get_rays_lean(
                directions_i_neighbor, poses_mtx_batched
            )  # both (b, 3)
            if args.ray_type == "ndc":
                rays_o_i_neighbor, rays_d_i_neighbor = ndc_rays_blender2(
                    H,
                    W,
                    [focal_refine, focal_refine],
                    1.0,
                    rays_o_i_neighbor,
                    rays_d_i_neighbor,
                )
            rays_train_i_neighbor = torch.cat(
                [rays_o_i_neighbor, rays_d_i_neighbor], -1
            ).view(-1, 6)
            directions_j_neighbor = get_ray_directions_lean(
                i, j_neighbor, [focal_refine, focal_refine], [W / 2, H / 2]
            )
            rays_o_j_neighbor, rays_d_j_neighbor = get_rays_lean(
                directions_j_neighbor, poses_mtx_batched
            )  # both (b, 3)
            if args.ray_type == "ndc":
                rays_o_j_neighbor, rays_d_j_neighbor = ndc_rays_blender2(
                    H,
                    W,
                    [focal_refine, focal_refine],
                    1.0,
                    rays_o_j_neighbor,
                    rays_d_j_neighbor,
                )
            rays_train_j_neighbor = torch.cat(
                [rays_o_j_neighbor, rays_d_j_neighbor], -1
            ).view(-1, 6)
            xyz_sampled_i_neighbor, z_vals_i_neighbor, ray_valid_i_neighbor = sampleXYZ(
                tensorf,
                rays_train_i_neighbor,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            (
                _,
                _,
                _,
                _,
                _,
                _,
                rgb_points_static_i_neighbor,
                sigmas_static_i_neighbor,
                _,
                _,
            ) = tensorf_static(
                rays_train_i_neighbor,
                ts_train,
                None,
                xyz_sampled_i_neighbor,
                z_vals_i_neighbor,
                ray_valid_i_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            (
                _,
                _,
                blending_i_neighbor,
                _,
                _,
                _,
                rgb_points_dynamic_i_neighbor,
                sigmas_dynamic_i_neighbor,
                z_vals_dynamic_i_neighbor,
                dists_dynamic_i_neighbor,
            ) = tensorf(
                rays_train_i_neighbor,
                ts_train,
                None,
                xyz_sampled_i_neighbor,
                z_vals_i_neighbor,
                ray_valid_i_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            _, _, _, _, _, depth_map_s_i_neighbor, _, _, _, _, _, _, _ = raw2outputs(
                rgb_points_static_i_neighbor,
                sigmas_static_i_neighbor,
                rgb_points_dynamic_i_neighbor,
                sigmas_dynamic_i_neighbor,
                dists_dynamic_i_neighbor,
                blending_i_neighbor,
                z_vals_dynamic_i_neighbor,
                rays_train_i_neighbor,
                is_train=True,
                ray_type=args.ray_type,
            )
            xyz_sampled_j_neighbor, z_vals_j_neighbor, ray_valid_j_neighbor = sampleXYZ(
                tensorf,
                rays_train_j_neighbor,
                N_samples=nSamples,
                ray_type=args.ray_type,
                is_train=True,
            )
            (
                _,
                _,
                _,
                _,
                _,
                _,
                rgb_points_static_j_neighbor,
                sigmas_static_j_neighbor,
                _,
                _,
            ) = tensorf_static(
                rays_train_j_neighbor,
                ts_train,
                None,
                xyz_sampled_j_neighbor,
                z_vals_j_neighbor,
                ray_valid_j_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            (
                _,
                _,
                blending_j_neighbor,
                _,
                _,
                _,
                rgb_points_dynamic_j_neighbor,
                sigmas_dynamic_j_neighbor,
                z_vals_dynamic_j_neighbor,
                dists_dynamic_j_neighbor,
            ) = tensorf(
                rays_train_j_neighbor,
                ts_train,
                None,
                xyz_sampled_j_neighbor,
                z_vals_j_neighbor,
                ray_valid_j_neighbor,
                is_train=True,
                white_bg=white_bg,
                ray_type=args.ray_type,
                N_samples=nSamples,
            )
            _, _, _, _, _, depth_map_s_j_neighbor, _, _, _, _, _, _, _ = raw2outputs(
                rgb_points_static_j_neighbor,
                sigmas_static_j_neighbor,
                rgb_points_dynamic_j_neighbor,
                sigmas_dynamic_j_neighbor,
                dists_dynamic_j_neighbor,
                blending_j_neighbor,
                z_vals_dynamic_j_neighbor,
                rays_train_j_neighbor,
                is_train=True,
                ray_type=args.ray_type,
            )
            disp_smooth_loss = torch.mean(
                (
                    (1.0 / torch.clamp(depth_map_s, min=1e-6))
                    - (1.0 / torch.clamp(depth_map_s_i_neighbor, min=1e-6))
                )
                ** 2
            ) + torch.mean(
                (
                    (1.0 / torch.clamp(depth_map_s, min=1e-6))
                    - (1.0 / torch.clamp(depth_map_s_j_neighbor, min=1e-6))
                )
                ** 2
            )
            total_loss += disp_smooth_loss * 50.0 * Temp_disp_TV
            summary_writer.add_scalar(
                "train/disp_smooth_loss",
                disp_smooth_loss.detach().item(),
                global_step=iteration,
            )

        if args.optimize_poses:
            optimizer_pose.zero_grad()
        if args.optimize_focal_length:
            optimizer_focal.zero_grad()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if args.optimize_poses:
            optimizer_pose.step()
            scheduler.step()
        if args.optimize_focal_length:
            optimizer_focal.step()
            scheduler_focal.step()

        pose_aligned = poses_mtx.clone().detach()

        summary_writer.add_scalar(
            "train/density_app_plane_lr",
            optimizer.param_groups[0]["lr"],
            global_step=iteration,
        )
        summary_writer.add_scalar(
            "train/basis_mat_lr", optimizer.param_groups[4]["lr"], global_step=iteration
        )
        if args.optimize_poses:
            summary_writer.add_scalar(
                "train/lr_pose",
                optimizer_pose.param_groups[0]["lr"],
                global_step=iteration,
            )
        if args.optimize_focal_length:
            summary_writer.add_scalar(
                "train/lr_focal",
                optimizer_focal.param_groups[0]["lr"],
                global_step=iteration,
            )

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            if args.with_GT_poses:
                pbar.set_description(
                    f"Iteration {iteration:05d}:"
                    + f" train_psnr = {float(np.mean(PSNRs)):.2f}"
                    + f" test_psnr = {float(np.mean(PSNRs_test)):.2f}"
                )
            else:
                pbar.set_description(f"Iteration {iteration:05d}:")
            PSNRs = []

            # matplotlib poses visualization
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            if args.with_GT_poses:
                vertices, faces, wireframe = get_camera_mesh(allposes, 0.005)
                center_gt = vertices[:, -1]
                ax.scatter(
                    center_gt[:, 0],
                    center_gt[:, 1],
                    center_gt[:, 2],
                    marker="o",
                    color="C0",
                )
                wireframe_merged = merge_wireframes(wireframe)
                for c in range(center_gt.shape[0]):
                    ax.plot(
                        wireframe_merged[0][c * 10 : (c + 1) * 10],
                        wireframe_merged[1][c * 10 : (c + 1) * 10],
                        wireframe_merged[2][c * 10 : (c + 1) * 10],
                        color="C0",
                    )

            vertices, faces, wireframe = get_camera_mesh(pose_aligned.cpu(), 0.005)
            center = vertices[:, -1]
            ax.scatter(center[:, 0], center[:, 1], center[:, 2], marker="o", color="C1")
            wireframe_merged = merge_wireframes(wireframe)
            for c in range(center.shape[0]):
                ax.plot(
                    wireframe_merged[0][c * 10 : (c + 1) * 10],
                    wireframe_merged[1][c * 10 : (c + 1) * 10],
                    wireframe_merged[2][c * 10 : (c + 1) * 10],
                    color="C1",
                )

            if args.with_GT_poses:
                for i in range(center.shape[0]):
                    ax.plot(
                        [center_gt[i, 0], center[i, 0]],
                        [center_gt[i, 1], center[i, 1]],
                        [center_gt[i, 2], center[i, 2]],
                        color="red",
                    )

            set_axes_equal(ax)
            plt.tight_layout()
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = (np.transpose(img, (2, 0, 1)) / 255.0).astype(np.float32)
            summary_writer.add_image("camera_poses", img, iteration)
            plt.close(fig)

            tensorf.save(
                poses_mtx.detach().cpu(),
                focal_refine.detach().cpu(),
                f"{logfolder}/{args.expname}.th",
            )
            tensorf_static.save(
                poses_mtx.detach().cpu(),
                focal_refine.detach().cpu(),
                f"{logfolder}/{args.expname}_static.th",
            )

        if (
            iteration % args.vis_train_every == args.vis_train_every - 1
            and args.N_vis != 0
        ):
            (
                rgb_maps_tb,
                depth_maps_tb,
                blending_maps_tb,
                gt_rgbs_tb,
                induced_flow_f_tb,
                induced_flow_b_tb,
                induced_flow_s_f_tb,
                induced_flow_s_b_tb,
                delta_xyz_tb,
                rgb_maps_s_tb,
                depth_maps_s_tb,
                rgb_maps_d_tb,
                depth_maps_d_tb,
                monodepth_tb,
            ) = render(
                test_dataset,
                poses_mtx,
                focal_refine.cpu(),
                tensorf_static,
                tensorf,
                args,
                renderer,
                None,
                N_vis=args.N_vis,
                prtx="",
                N_samples=nSamples,
                white_bg=white_bg,
                ray_type=args.ray_type,
                compute_extra_metrics=False,
            )
            summary_writer.add_images(
                "test/rgb_maps",
                torch.stack(rgb_maps_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/rgb_maps_s",
                torch.stack(rgb_maps_s_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/rgb_maps_d",
                torch.stack(rgb_maps_d_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/depth_map",
                torch.stack(depth_maps_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/depth_map_s",
                torch.stack(depth_maps_s_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/depth_map_d",
                torch.stack(depth_maps_d_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/monodepth_tb",
                torch.stack(monodepth_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/blending_maps",
                torch.stack(blending_maps_tb, 0),
                global_step=iteration,
                dataformats="NCHW",
            )
            summary_writer.add_images(
                "test/gt_maps",
                torch.stack(gt_rgbs_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_f",
                torch.stack(induced_flow_f_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_b",
                torch.stack(induced_flow_b_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_s_f",
                torch.stack(induced_flow_s_f_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/induced_flow_s_b",
                torch.stack(induced_flow_s_b_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            # visualize the index 1 (has both forward and backward)
            gt_flow_f_tb_list = []
            gt_flow_b_tb_list = []
            allflows_f_ = allflows_f.view(args.N_voxel_t, H, W, 2)
            allflows_b_ = allflows_b.view(args.N_voxel_t, H, W, 2)
            for gt_flo_f, gt_flo_b in zip(allflows_f_, allflows_b_):
                gt_flow_f_tb_list.append(
                    torch.from_numpy(flow_to_image(gt_flo_f.detach().cpu().numpy()))
                )
                gt_flow_b_tb_list.append(
                    torch.from_numpy(flow_to_image(gt_flo_b.detach().cpu().numpy()))
                )
            summary_writer.add_images(
                "test/gt_flow_f",
                torch.stack(gt_flow_f_tb_list, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/gt_flow_b",
                torch.stack(gt_flow_b_tb_list, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            summary_writer.add_images(
                "test/delta_xyz_tb",
                torch.stack(delta_xyz_tb, 0),
                global_step=iteration,
                dataformats="NHWC",
            )
            gt_mask_tb_list = []
            allforegroundmasks_ = allforegroundmasks.view(args.N_voxel_t, H, W, 3)
            for foregroundmask in allforegroundmasks_:
                gt_mask_tb_list.append(foregroundmask)
            summary_writer.add_images(
                "test/gt_blending_maps",
                torch.stack(gt_mask_tb_list, 0),
                global_step=iteration,
                dataformats="NHWC",
            )

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)
            tensorf_static.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
                if args.optimize_poses:
                    optimizer_pose.param_groups[0]["lr"] = lr_pose
                if iteration >= args.upsamp_list[3] and args.optimize_focal_length:
                    optimizer_focal.param_groups[0]["lr"] = lr_pose
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf_static.get_optparam_groups(
                args.lr_init * lr_scale, args.lr_basis * lr_scale
            )
            grad_vars.extend(
                tensorf.get_optparam_groups(
                    args.lr_init * lr_scale, args.lr_basis * lr_scale
                )
            )
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration > args.n_iters // 2:
            optimizer_pose.param_groups[0]["lr"] = 0.0
            optimizer_focal.param_groups[0]["lr"] = 0.0

    tensorf.save(
        poses_mtx.detach().cpu(),
        focal_refine.detach().cpu(),
        f"{logfolder}/{args.expname}.th",
    )
    tensorf_static.save(
        poses_mtx.detach().cpu(),
        focal_refine.detach().cpu(),
        f"{logfolder}/{args.expname}_static.th",
    )

    os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
    PSNRs_train, near_fars, _ = evaluation(
        test_dataset,
        poses_mtx,
        focal_refine.cpu(),
        tensorf_static,
        tensorf,
        args,
        renderer,
        f"{logfolder}/imgs_test_all",
        N_vis=-1,
        N_samples=-1,
        white_bg=white_bg,
        ray_type=args.ray_type,
        device=device,
    )
    print(
        f"======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================"
    )
    # save poses_bounds.npy
    poses_saving = poses_mtx.clone()
    poses_saving = torch.cat(
        [-poses_saving[..., 1:2], poses_saving[..., :1], poses_saving[..., 2:4]], -1
    )
    hwf = (
        torch.from_numpy(np.array([H, W, focal_refine.detach().cpu()]))
        * args.downsample_train
    )
    hwf = torch.stack([hwf] * args.N_voxel_t, 0)[..., None]
    poses_saving = torch.cat([poses_saving.cpu(), hwf], -1).view(args.N_voxel_t, -1)
    poses_bounds_saving = (
        torch.cat([poses_saving, torch.from_numpy(np.array(near_fars))], -1)
        .detach()
        .numpy()
    )
    np.save(os.path.join(args.datadir, "poses_bounds_RoDynRF.npy"), poses_bounds_saving)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args, os.path.join(args.basedir, args.expname))
    else:
        reconstruction(args)
