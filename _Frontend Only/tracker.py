import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric


import torch
import time
from vggt.models.vggt import VGGT
import einops
import lietorch
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        
        self.reset_idx_f2k()


    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        # 현재 프레임과 마지막 키프레임 간 모델 추론 수행
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        ) 

        """
        idx_f2k: 현재 프레임 포인트 → 키프레임 포인트 매칭 인덱스
        Xff, Xkf: 두 프레임에서 매칭된 3D 포인트들
        Cff, Ckf: confidence
        Qff, Qkf: descriptor confidence (정합성)
        """
        
        
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )
        
    def track_init(self, frame):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images = frame.img.unsqueeze(0)
                print(images.shape)
                #images = torch.cat([frame.img], dim=0) # (2,3,H,W)
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(images)
                #aggregated_tokens_list, ps_idx = self.vggt.aggregator(frame.img)
            point_map, point_conf = self.vggt.point_head(aggregated_tokens_list, images, ps_idx)
                
        Xff_reduced = point_map[:, 0, :, :, :]

        # predictions["world_points_conf"]: (1, 3, H, W)
        # 첫 번째 요소 선택 → (1, H, W)
        Cff_reduced = point_conf[:, 0, :, :]
        
        Xff = einops.rearrange(Xff_reduced, "b h w c -> b (h w) c")  # 결과: (1, H*W, 3)
        Cff = einops.rearrange(Cff_reduced, "b h w -> b (h w) 1")   # 결과: (1, H*W, 1)
        return Xff, Cff
    
    '''
    def cam_param_to_sim3(self, param_tensor):
        """
        param_tensor: shape (9,) 텐서, 구성: [x, y, z, q1, q2, q3, q4, I1, I2]
        """
        # 1. Translation 추출
        t = param_tensor[:3]
        
        # 2. Quaternion 추출 후 정규화
        q = param_tensor[3:7]
        q = q / q.norm()  # 단위 quaternion
        
        # 3. 내부 파라미터에서 scale 결정 (여기서는 첫 번째 intrinsics를 scale로 사용)
        scale = 1
        
        # 4. lietorch.Sim3 객체 생성
        sim3 = lietorch.Sim3(translation=t, rotation=q, scale=scale)
        return sim3
    '''
        
    def cam_param_to_sim3(self, param_tensor):
        """
        param_tensor: (9,) = [tx, ty, tz, qx, qy, qz, qw, I1, I2]
        """
        device, dtype = param_tensor.device, param_tensor.dtype

        # 1) translation
        t = param_tensor[:3]

        # 2) quaternion → 단위 정규화
        q = param_tensor[3:7]
        q = q / (q.norm() + 1e-9)

        # 2-1) quat → so(3) axis-angle
        so3  = lietorch.SO3(q.unsqueeze(0))      # (1,4)
        rvec = so3.log().squeeze(0)              # (3,)

        # 3) log-scale σ (scale = 1 → σ = 0)
        log_s = torch.zeros(1, device=device, dtype=dtype)

        # 4) 7-D 벡터  [rx, ry, rz, tx, ty, tz, σ]
        sim_vec = torch.cat([rvec, t, log_s], dim=0)  # (7,)

        # 5) Sim3 생성  batch 차원 유지!
        sim3 = lietorch.Sim3.exp(sim_vec.unsqueeze(0))  # shape == (1,)

        return sim3          # squeeze(0) 하지 않음
    

    def track_vggt(self, frame: Frame, device):
        keyframe = self.keyframes.last_keyframe()

        idx_f2k, _, _, _, _, _, _, _ = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )

        """
        idx_f2k: 현재 프레임 포인트 → 키프레임 포인트 매칭 인덱스
        Xff, Xkf: 두 프레임에서 매칭된 3D 포인트들
        Cff, Ckf: confidence
        Qff, Qkf: descriptor confidence (정합성)
        """
        
        # images = load_and_preprocess_images_cv2([frame.img, keyframe.img]) 필요 없을 듯?
        #print(frame.img.shape, keyframe.img.shape)
        images = torch.cat([frame.img, keyframe.img.unsqueeze(0)], dim=0) # (2,3,H,W)
        
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images = images[None]
                aggregated_tokens_list, ps_idx = self.vggt.aggregator(images)
                
                
            pose_enc = self.vggt.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            #extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            #depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

            # Predict Point Maps
            point_map, point_conf = self.vggt.point_head(aggregated_tokens_list, images, ps_idx)
                
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            #point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
            #                                                            extrinsic.squeeze(0), 
            #                                                            intrinsic.squeeze(0))     
            
            H = frame.img.shape[2]
            W = frame.img.shape[3]

            # meshgrid를 이용해 모든 픽셀 좌표 (x, y)를 생성 (x: 열, y: 행)
            stride = 4
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, stride, device=device),
                torch.arange(0, W, stride, device=device),
                indexing='ij'
            )
            #grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            # 두 grid를 스택하여 (H, W, 2) 형태의 텐서를 만든 후, (H*W, 2) 형태로 reshape
            all_query_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

            # batch 차원을 추가 (예: (1, H*W, 2))
            all_query_points = all_query_points[None]

            # 이후 기존 코드와 같이 model.track_head를 호출합니다.
            track_list, vis_score, conf_score = self.vggt.track_head(aggregated_tokens_list, images, ps_idx, query_points=all_query_points)
            #2개의 이미지에서 3개의 query_points 일때 (1,2,3,2)형태이다. 첫번째 2는 2개의 이미지, 가운데 3은 3개의 query, 맨마지막 2는 x,y 좌표이다
            #conf score은 (1,2,3) 형태이다. 2는 2개의 이미지, 3은 3개의 query이다.
            track_list = track_list[-1][-1]
            conf_score = conf_score[-1][1:]
            
       

        # Update keyframe pointmap after registration (need pose)
        # predictions["world_points"] 형식은 (1, 3, H, W, 3)
        # predictions["world_points_conf"] 형식은 (1, 3, H, W)
        
        Xff_reduced = point_map[:, 0, :, :, :]

        # predictions["world_points_conf"]: (1, 3, H, W)
        # 첫 번째 요소 선택 → (1, H, W)
        Cff_reduced = point_conf[:, 0, :, :]

        # 시스템이 기대하는 형식으로 재배열:
        Xff = einops.rearrange(Xff_reduced, "b h w c -> b (h w) c")  # 결과: (1, H*W, 3)
        Cff = einops.rearrange(Cff_reduced, "b h w -> b (h w) 1")   # 결과: (1, H*W, 1)

        # 가정이다. Xff 는 자기 좌표계에서의 3D pointmap이다. 이때 자기 좌표계는 0,0,0,1, 0,0,0 이다.
        frame.update_pointmap(Xff, Cff)

        
        valid_matches = conf_score > self.cfg["C_conf"]  # shape: (2, 3)

        # 전체 유효 매칭 비율 계산
        match_frac = valid_matches.float().mean()  # 모든 이미지와 query에 대한 평균
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        
                
        # 카메로 포즈 자체가 현재 프레임 -> 월드 좌표
        T_CkCf = self.cam_param_to_sim3(pose_enc[0,1])
        #print(T_CkCf.data.shape)
        frame.T_WC =   keyframe.T_WC * T_CkCf
        # T_WCf = T_WCk * T_CkCf
        # frame.T_WC = T_WCf
        
        
        Xkf_reduced = point_map[:, 1, :, :, :]

        # predictions["world_points_conf"]: (1, 3, H, W)
        # 첫 번째 요소 선택 → (1, H, W)
        Ckf_reduced = point_conf[:, 1, :, :]

        # 시스템이 기대하는 형식으로 재배열:
        Xkf = einops.rearrange(Xkf_reduced, "b h w c -> b (h w) c")  # 결과: (1, H*W, 3)
        Ckf = einops.rearrange(Ckf_reduced, "b h w -> b (h w) 1")   # 결과: (1, H*W, 1)

        #print(Xkf.shape)
        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf.squeeze(0)).unsqueeze(0)
        #print(Xkf.squeeze(0).shape)
        #Xkk = T_CkCf.act(Xkf.unsqueeze(0)).squeeze(0)   # (N, 3)
        
        keyframe.update_pointmap(Xkk, Ckf)
        
        
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        """
        ToDo
        
        vggt tracking 활성화 및 무엇을 반환하는지 확인
        결국 Matching 정보를 활용해야하는데 vggt을 통해 matching 정보를 어떻게 얻을지 확인
        
        """
        valid_kf = valid_matches[0]  # shape: (3,) → 키프레임의 3개 query에 대해 True/False

        # 유효 매칭 수와 비율
        n_valid = valid_kf.float().sum()
        match_frac_k = n_valid / valid_kf.numel()  # 키프레임 쪽의 유효한 매칭 비율

        # 고유한 키프레임 픽셀의 수를 계산합니다.
        # track_list의 shape는 (2, 3, 2)에서, 키프레임 좌표는 track_list[1] (shape: (3, 2))
        keyframe_points = track_list[1]  # 각 query의 (x,y) 좌표

        # 유효한 매칭에 해당하는 키프레임 좌표 선택
        valid_keyframe_points = keyframe_points[valid_kf]  # (N_valid, 2)

        # 픽셀 좌표는 소수점 값을 가질 수 있으므로, 정수 단위(반올림 후)를 사용해 고유 픽셀 수를 계산
        unique_pixels = torch.unique(valid_keyframe_points.round(), dim=0)
        unique_frac_f = unique_pixels.shape[0] / valid_kf.numel()

        # 최종적으로, 두 비율 중 낮은 값이 기준(self.cfg["match_frac_thresh"]) 미만이면 새로운 keyframe을 선택
        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]


        # # Keyframe selection
        # n_valid = valid_kf.sum()
        # match_frac_k = n_valid / valid_kf.numel()
        # unique_frac_f = (
        #     torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        # )

        # new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                #Qkf,
                #Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
