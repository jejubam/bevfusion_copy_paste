
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.datasets import GlobalRotScaleTrans
from mmdet3d.registry import TRANSFORMS

from typing import List, Tuple, Union, Dict, Any, Optional
import random, pickle, os

MapType = Dict[int, List[Tuple[str, list]]]

@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip,
                 is_train):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_matrix'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVFusionRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('horizontal')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('horizontal')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, :, ::-1].copy()

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in data:
                data['points'].flip('vertical')
            if 'gt_bboxes_3d' in data:
                data['gt_bboxes_3d'].flip('vertical')
            if 'gt_masks_bev' in data:
                data['gt_masks_bev'] = data['gt_masks_bev'][:, ::-1, :].copy()

        if 'lidar_aug_matrix' not in data:
            data['lidar_aug_matrix'] = np.eye(4)
        data['lidar_aug_matrix'][:3, :] = rotation @ data[
            'lidar_aug_matrix'][:3, :]
        return data


@TRANSFORMS.register_module()
class BEVFusionGlobalRotScaleTrans(GlobalRotScaleTrans):
    """Compared with `GlobalRotScaleTrans`, the augmentation order in this
    class is rotation, translation and scaling (RTS)."""

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._trans_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'T', 'S'])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = input_dict['pcd_rotation'].T * input_dict[
            'pcd_scale_factor']
        lidar_augs[:3, 3] = input_dict['pcd_trans'] * \
            input_dict['pcd_scale_factor']

        if 'lidar_aug_matrix' not in input_dict:
            input_dict['lidar_aug_matrix'] = np.eye(4)
        input_dict[
            'lidar_aug_matrix'] = lidar_augs @ input_dict['lidar_aug_matrix']

        return input_dict


@TRANSFORMS.register_module()
class GridMask(BaseTransform):

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def transform(self, results):
        if np.random.rand() > self.prob:
            return results
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask

        # mask = mask.expand_as(imgs[0])
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            imgs = [x * mask + offset for x in imgs]
        else:
            imgs = [x * mask for x in imgs]

        results.update(img=imgs)
        return results

from mmdet3d.structures.points import BasePoints
from mmdet3d.structures.ops import box_np_ops
import json
import os

# TODO Ped ObjectSample
@TRANSFORMS.register_module()
class PedObjectSample(BaseTransform):

    def __init__(self, db_sampler, log_dir='logs', image_size=(1600,900),
                 cam_name='CAM_FRONT', sample_2d=False, use_ground_plane=False):
        super().__init__()
        self.sampler_cfg = db_sampler
        if 'type' not in db_sampler:
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = TRANSFORMS.build(db_sampler)
        self.sample_2d = sample_2d
        self.use_ground_plane = use_ground_plane
        self.disabled = False
        self.image_size = image_size
        self.cam_name = cam_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.epoch = 0  # 외부에서 세팅
        self.log_path = os.path.join(self.log_dir, "pedobject_epoch_00.jsonl")

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.log_path = os.path.join(self.log_dir, f"pedobject_epoch_{epoch:02d}.jsonl")

    def inside_cam_fov(self, boxes3d_lidar, lidar2cam, cam_intrinsic, image_size):
        """
        boxes3d_lidar: (N, 7) [x,y,z,w,l,h,yaw] (LiDAR 좌표계)
        lidar2cam: (4, 4)
        cam_intrinsic: (3,3) or (3,4) or (4,4)
        image_size: (W, H)
        returns: mask (N,), uv (N,2), z (N,)
        """
        boxes3d_lidar = np.asarray(boxes3d_lidar)
        lidar2cam = np.asarray(lidar2cam)
        K = np.asarray(cam_intrinsic)

        N = boxes3d_lidar.shape[0]
        centers_h = np.concatenate([boxes3d_lidar[:, :3], np.ones((N, 1))], axis=1)  # (N,4)
        Xc = (lidar2cam @ centers_h.T).T  # (N,4) camera coords (homogeneous)
        z = Xc[:, 2]

        # 프로젝션 행렬 P 구성
        # - K가 3x3이면 P = [K | 0] (3x4)
        # - K가 3x4면 그대로 사용
        # - K가 4x4면 P = K[:3,:4]
        if K.shape == (3, 3):
            P = np.concatenate([K, np.zeros((3, 1), dtype=K.dtype)], axis=1)  # (3,4)
        elif K.shape[0] == 3 and K.shape[1] == 4:
            P = K[:3, :4]
        else:
            # 4x4 같은 케이스
            P = K[:3, :4]

        # 동차투영
        uvw = (P @ Xc.T).T  # (N,3)
        w = np.clip(uvw[:, 2:3], 1e-6, None)
        uv = uvw[:, :2] / w  # (N,2)

        W, H = image_size  # 주의: (W,H) 순서로 넣자
        mask = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        
        return mask
    
    @staticmethod
    def remove_points_in_boxes(points: BasePoints,
                               boxes: np.ndarray) -> np.ndarray:
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points
    
    def transform(self, results):
        if self.disabled:
            return results

        # CAM_FRONT_INDEX = 0
        lidar2cam = results['lidar2cam'][0]
        cam_intrinsic = results['cam2img'][0]

        # pedestrian만 추출
        sampled = self.db_sampler.sample_one(
            results['gt_bboxes_3d'].numpy(),
            results['gt_labels_3d'],
            img=None)
        if sampled is None:
            return results
        
        boxes3d = sampled['gt_bboxes_3d']
        valid_mask = self.inside_cam_fov(boxes3d, lidar2cam, cam_intrinsic, self.image_size)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return results

        # 하나만 선택
        idx = np.random.choice(valid_indices)
        sel_box = boxes3d[idx:idx+1]
        start, end = sampled['obj_slices'][idx]     
        sel_points = sampled['points'][start:end] 
        sel_label = sampled['gt_labels_3d'][idx:idx+1]

        # 기존 points/bbox/label 업데이트
        points = self.remove_points_in_boxes(results['points'], sel_box)
        results['points'] = points.cat([sel_points, points])
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'].new_box(
            np.concatenate([results['gt_bboxes_3d'].numpy(), sel_box]))
        results['gt_labels_3d'] = np.concatenate(
            [results['gt_labels_3d'], sel_label], axis=0)

        # json 로그 저장
        log_item = {
            "epoch": self.epoch,
            "img_path": results.get("img_path", None),
            "sample_idx": results.get("sample_idx", None),
            "added_object": {
                "class": "pedestrian",
                "db_path": sampled['db_infos'][idx]['path'],
                "box3d_lidar": sel_box.tolist()
            }
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_item) + '\n')
        
        # points_np = results['points'].tensor.numpy().astype(np.float32)
        # boxes_np = results['gt_bboxes_3d'].tensor.numpy().astype(np.float32)
        # base_name = f"epoch_{self.epoch:02d}_{results['sample_idx']}"
        # np.save(f"{self.log_dir}/{base_name}_points.npy", points_np)
        # np.save(f"{self.log_dir}/{base_name}_bboxes.npy", boxes_np)
        
        # # or binary format
        # points_np.tofile(f"{self.log_dir}/{base_name}.bin")

        return results

@TRANSFORMS.register_module()
class CutAndPaste(BaseTransform):
    def __init__(self,
                 db_sampler,
                 log_dir_path='logs',
                 image_size=(1600, 900),
                 cam_name='CAM_FRONT',
                 data_info_path=None):

        super().__init__()

        # db_sampler 준비
        if 'type' not in db_sampler:
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = TRANSFORMS.build(db_sampler)

        self.disabled = False
        self.cam_name = cam_name
        self.image_size = image_size
        self.log_dir = log_dir_path
        self.epoch = 0
        self.log_path = os.path.join(self.log_dir, "pedobject_epoch_00.jsonl")

        os.makedirs(self.log_dir, exist_ok=True)

        # log 매핑 캐시
        self._map_cache = None
        self._map_src_mtime = None

        # data_info는 init에서 1회만 로드 → (중요)
        assert data_info_path is not None
        with open(data_info_path, "rb") as f:
            self.data_info = pickle.load(f)

    # ---------------------------------------------------------
    # Log 매핑 캐시
    # ---------------------------------------------------------
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.log_path = os.path.join(
            self.log_dir, f"pedobject_epoch_{epoch:02d}.jsonl")
        self._map_cache = None
        self._map_src_mtime = None

    def _ensure_map_cache(self):
        jp = self.log_path
        if not os.path.exists(jp):
            self._map_cache = {}
            return

        src_m = os.path.getmtime(jp)
        if self._map_cache is not None and self._map_src_mtime == src_m:
            return

        mapping = {}
        with open(jp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sidx = int(rec.get("sample_idx"))
                    added = rec.get("added_object", {})
                    dbp = added.get("db_path")
                    box3d = added.get("box3d_lidar")
                    if dbp is None or box3d is None:
                        continue
                    mapping.setdefault(sidx, []).append((dbp, box3d))
                except:
                    continue

        self._map_cache = mapping
        self._map_src_mtime = src_m

    def _lookup(self, sample_idx):
        self._ensure_map_cache()
        hits = self._map_cache.get(int(sample_idx), None)
        if not hits:
            return None
        return hits[0] if len(hits) == 1 else hits

    # ---------------------------------------------------------
    # Utility : remove points inside 3D box
    # ---------------------------------------------------------
    @staticmethod
    def remove_points_in_boxes(points: BasePoints, boxes: np.ndarray):
        mask = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(mask.any(-1))]
        return points

    # ---------------------------------------------------------
    # get 8 corners of 3D box
    # ---------------------------------------------------------
    @staticmethod
    def get_corners(box):
        cx, cy, z0, dx, dy, dz, yaw = box[:7]
        z1 = z0 + dz
        xh, yh = dx / 2, dy / 2

        base = np.array([
            [xh,  yh, z0],
            [xh, -yh, z0],
            [-xh, -yh, z0],
            [-xh,  yh, z0],
            [xh,  yh, z1],
            [xh, -yh, z1],
            [-xh, -yh, z1],
            [-xh,  yh, z1]
        ])

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s],
                      [s,  c]])
        xy = base[:, :2] @ R.T
        xy[:, 0] += cx
        xy[:, 1] += cy

        out = np.zeros_like(base)
        out[:, :2] = xy
        out[:, 2] = base[:, 2]
        return out

    # ---------------------------------------------------------
    # Projection
    # ---------------------------------------------------------
    @staticmethod
    def project(corners, lidar2cam, K):
        homo = np.concatenate([corners, np.ones((8, 1))], axis=1)
        cam = (lidar2cam @ homo.T).T
        X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
        valid = Z > 1e-6

        uvw = K @ np.stack([X, Y, Z], axis=0)
        u = uvw[0] / (uvw[2] + 1e-12)
        v = uvw[1] / (uvw[2] + 1e-12)

        u = np.where(valid, u, np.nan)
        v = np.where(valid, v, np.nan)
        return u, v

    # ---------------------------------------------------------
    # bbox clamp
    # ---------------------------------------------------------
    @staticmethod
    def clamp(x1, y1, x2, y2, W, H):
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        return int(x1), int(y1), int(x2), int(y2)

    # ---------------------------------------------------------
    # feather blending
    # ---------------------------------------------------------
    @staticmethod
    def feather_blend(target_img, patch, x1, y1, radius=10):
        tgt_np = np.array(target_img).astype(np.float32)
        patch_np = np.array(patch).astype(np.float32)

        h, w = patch_np.shape[:2]

        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.minimum(np.minimum(xx, w - 1 - xx),
                          np.minimum(yy, h - 1 - yy))
        mask = np.clip(dist / float(radius), 0.0, 1.0)[..., None]

        out = tgt_np.copy()
        region = out[y1:y1 + h, x1:x1 + w, :]
        blended = region * (1 - mask) + patch_np * mask
        out[y1:y1 + h, x1:x1 + w, :] = blended
        return Image.fromarray(out.astype(np.uint8))

    # ---------------------------------------------------------
    # transform() : 최종 augmentation
    # ---------------------------------------------------------
    def transform(self, results):

        if self.disabled:
            return results

        # 1) log lookup → db_path, box3d 가져오기
        sidx = results.get("sample_idx", None)
        hit = self._lookup(sidx)
        if hit is None:
            return results

        if isinstance(hit, list):
            db_path, box3d = random.choice(hit)
        else:
            db_path, box3d = hit

        # sample from db
        sampled = self.db_sampler.sample_idx(db_path)
        if sampled is None:
            return results

        sel_box = sampled['gt_bboxes_3d']   # (1,7)
        sel_points = sampled['points']
        sel_label = sampled['gt_labels_3d']

        # 2) pointcloud 업데이트
        pts = self.remove_points_in_boxes(results['points'], sel_box)
        results['points'] = pts.cat([sel_points, pts])
        results['gt_bboxes_3d'] = results['gt_bboxes_3d'].new_box(
            np.concatenate([results['gt_bboxes_3d'].numpy(), sel_box])
        )
        results['gt_labels_3d'] = np.concatenate(
            [results['gt_labels_3d'], sel_label], axis=0
        )

        # -----------------------------------------------------
        # 3) Image Cut-Paste
        # -----------------------------------------------------

        # 3-1) Source 이미지 찾기
        src_sample_idx = int(db_path.split('/')[1].split('_')[0])
        src_info = self.data_info['data_list'][src_sample_idx]['images'][self.cam_name]

        src_path = os.path.join(
            "/home/donguk/mmdetection3d/data/nuscenes/samples",
            self.cam_name, src_info['img_path']
        )
        src_img = Image.open(src_path).convert("RGB")
        src_W, src_H = src_img.size

        K_S = np.array(src_info['cam2img'])
        l2c_S = np.array(src_info['lidar2cam'])

        # 3-2) Target 이미지(results)
        tgt_np = results['img'][0]  # numpy
        tgt_img = Image.fromarray(tgt_np)
        tgt_W, tgt_H = tgt_img.size

        tgt_info = self.data_info['data_list'][sidx]['images'][self.cam_name]
        K_T = np.array(tgt_info['cam2img'])
        l2c_T = np.array(tgt_info['lidar2cam'])

        # 3D corners
        corners = self.get_corners(np.array(box3d[0]))

        # 3-3) target bbox
        uT, vT = self.project(corners, l2c_T, K_T)
        finite = np.isfinite(uT) & np.isfinite(vT)
        x1T, y1T = uT[finite].min(), vT[finite].min()
        x2T, y2T = uT[finite].max(), vT[finite].max()
        x1T, y1T, x2T, y2T = self.clamp(x1T, y1T, x2T, y2T, tgt_W, tgt_H)

        # 3-4) source bbox
        uS, vS = self.project(corners, l2c_S, K_S)
        finite = np.isfinite(uS) & np.isfinite(vS)
        x1S, y1S = uS[finite].min(), vS[finite].min()
        x2S, y2S = uS[finite].max(), vS[finite].max()
        x1S, y1S, x2S, y2S = self.clamp(x1S, y1S, x2S, y2S, src_W, src_H)

        # crop + resize
        patch = src_img.crop((x1S, y1S, x2S, y2S))
        Tw, Th = x2T - x1T, y2T - y1T
        patch_resized = patch.resize((Tw, Th))

        # feather blending
        tgt_out = self.feather_blend(tgt_img, patch_resized, x1T, y1T, radius=10)

        # numpy로 되돌리기
        results['img'][0] = np.array(tgt_out)

        return results
