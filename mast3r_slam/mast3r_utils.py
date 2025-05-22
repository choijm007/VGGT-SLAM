import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q

@torch.inference_mode
def vggt_symmetric_inference(model, device, frame_i, frame_j):
    
    images1 = torch.cat([frame_i.img, frame_j.img.unsqueeze(0)], dim=0) # (2,3,H,W)
    images2 = torch.cat([frame_j.img, frame_i.img.unsqueeze(0)], dim=0) # (2,3,H,W)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images1)
            aggregated_tokens_list2, ps_idx2 = model.aggregator(images2)
        # (B, S, C, H//2, W//2)
        D,Q = model.track_head.feature_extractor(aggregated_tokens_list, images, ps_idx)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))     
        
        point_map = point_map_by_unprojection[None, ...]
        if point_map.dtype != np.float32:
            point_map = point_map.astype(np.float32)
            point_map = torch.from_numpy(point_map)
            point_map = point_map.to(device)
            
        X1 = point_map.squeeze(0)
        C1 = depth_conf.squeeze(0)
        D1 = D.squeeze(0)
        Q1 = Q.squeeze(0)
        
        D,Q = model.track_head.feature_extractor(aggregated_tokens_list2, images2, ps_idx2)
        pose_enc = model.camera_head(aggregated_tokens_list2)[-1]
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list2, images2, ps_idx2)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images2.shape[-2:])
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))     
        
        point_map = point_map_by_unprojection[None, ...]
        if point_map.dtype != np.float32:
            point_map = point_map.astype(np.float32)
            point_map = torch.from_numpy(point_map)
            point_map = point_map.to(device)
            
        X2 = point_map.squeeze(0)
        C2 = depth_conf.squeeze(0)
        D2 = D.squeeze(0)
        Q2 = Q.squeeze(0)
        
        
    X = torch.cat([X1, X2], dim=0)
    C = torch.cat([C1, C2], dim=0)
    D = torch.cat([D1, D2], dim=0)
    Q = torch.cat([Q1, Q2], dim=0)
        
    
    return X, C, D, Q
    


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def vggt_decode_symmetric_batch(
    model, device, frame_i, frame_j, shape_i, shape_j
):
    #print(frame_i)
    B = frame_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        #print(frame_i[b].shape)
        images1 = torch.cat([frame_i[b][None], frame_j[b][None]], dim=0) # (2,3,H,W)
        images2 = torch.cat([frame_j[b][None], frame_i[b][None]], dim=0) # (2,3,H,W)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                images1 = images1[None]
                images2 = images2[None]
                #print(images1.shape)
                aggregated_tokens_list, ps_idx = model.aggregator(images1)
                aggregated_tokens_list2, ps_idx2 = model.aggregator(images2)
            # (B, S, C, H//2, W//2)
            D1 = model.track_head.feature_extractor(aggregated_tokens_list, images1, ps_idx)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images1, ps_idx)
            
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images1.shape[-2:])
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                        extrinsic.squeeze(0), 
                                                                        intrinsic.squeeze(0))     
            
            point_map = point_map_by_unprojection[None, ...]
            if point_map.dtype != np.float32:
                point_map = point_map.astype(np.float32)
                point_map = torch.from_numpy(point_map)
                point_map = point_map.to(device)
                
            X1 = point_map.squeeze(0)
            C1 = depth_conf.squeeze(0)
            D1 = D1.squeeze(0)
            D1 = D1.permute(0,2,3,1)
            #Q1 = Q.squeeze(0)
            
            D2 = model.track_head.feature_extractor(aggregated_tokens_list2, images2, ps_idx2)
            pose_enc = model.camera_head(aggregated_tokens_list2)[-1]
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list2, images2, ps_idx2)
            
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images2.shape[-2:])
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                        extrinsic.squeeze(0), 
                                                                        intrinsic.squeeze(0))     
            
            point_map = point_map_by_unprojection[None, ...]
            if point_map.dtype != np.float32:
                point_map = point_map.astype(np.float32)
                point_map = torch.from_numpy(point_map)
                point_map = point_map.to(device)
                
            X2 = point_map.squeeze(0)
            C2 = depth_conf.squeeze(0)
            D2 = D2.squeeze(0)
            D2 = D2.permute(0,2,3,1)
            # Q2 = Q.squeeze(0)
            
        
        X.append(torch.cat([X1, X2], dim=0))
        C.append(torch.cat([C1, C2], dim=0))
        D.append(torch.cat([D1, D2], dim=0))
        #Q.append(torch.cat([Q1, Q2], dim=0))
        
    
    
        

    X, C, D = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        #torch.stack(Q, dim=1),
    )
    # X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D



@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii



@torch.inference_mode
def vggt_inference_mono(model, device,frame):
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = frame.img.unsqueeze(0)
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        # (B, S, C, H//2, W//2)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))     
        
        point_map = point_map_by_unprojection[None, ...]
        if point_map.dtype != np.float32:
            point_map = point_map.astype(np.float32)
            point_map = torch.from_numpy(point_map)
            point_map = point_map.to(device)
            
    X = point_map.squeeze(0)
    C = depth_conf.squeeze(0)
    
    Xff = einops.rearrange(X, "b h w c -> b (h w) c")  # 결과: (1, H*W, 3)
    Cff = einops.rearrange(C, "b h w -> b (h w) 1")   # 결과: (1, H*W, 1)
    return Xff, Cff


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )

def vggt_match_symmetric(model,vggt, device, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j,frame_i, frame_j,):
    X, C, D = vggt_decode_symmetric_batch(vggt, device, frame_i, frame_j, shape_i, shape_j)
    _,_,_,Q = mast3r_decode_symmetric_batch(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j)
    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q

@torch.inference_mode
def vggt_asymmetric_inference(model, device,frame_i, frame_j):

    
    '''
    왜 unsqueeze(0) 이 되는건지 파악
    '''
    images = torch.cat([frame_i.img, frame_j.img.unsqueeze(0)], dim=0) # (2,3,H,W)
    # print(frame_i.img.shape, frame_j.img.shape)
    # print(images.shape)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        # (B, S, C, H//2, W//2)
        D= model.track_head.feature_extractor(aggregated_tokens_list, images, ps_idx)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))     
        
        point_map = point_map_by_unprojection[None, ...]
        if point_map.dtype != np.float32:
            point_map = point_map.astype(np.float32)
            point_map = torch.from_numpy(point_map)
            point_map = point_map.to(device)
            
    X = point_map.squeeze(0)
    C = depth_conf.squeeze(0)
    D = D.squeeze(0)
    D = D.permute(0,2,3,1)
    
    #print(D.shape)
    return X, C, D,  pose_enc



def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):

    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji

def vggt_match_asymmetric(model, vggt, device,frame_i, frame_j, idx_i2j_init=None):
    X, C, D, pose_enc = vggt_asymmetric_inference(vggt,device, frame_i, frame_j)
    _,_,_,Q = mast3r_asymmetric_inference(model, frame_i, frame_j)
    # print(X.shape, C.shape, D.shape)
    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]
    
    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
    
    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, pose_enc




def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size=448, square_ok=False, return_transformation=False):
    # size는 112의 배수여야 함
    #print(size)
    assert size % 112 == 0, "size must be a multiple of 112"
    
    # numpy → PIL
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size

    # 리사이즈: 긴 변이 size가 되도록
    if W1 >= H1:
        new_W = size
        new_H = int(round(H1 * size / W1))
    else:
        new_H = size
        new_W = int(round(W1 * size / H1))
    img = img.resize((new_W, new_H), PIL.Image.BILINEAR)

    # 중앙 크롭: 가로/세로를 112의 배수로 맞춤
    crop_w = (new_W // 112) * 112
    crop_h = (new_H // 112) * 112

    if square_ok:
        side = min(crop_w, crop_h)
        crop_w = crop_h = side

    left = (new_W - crop_w) // 2
    top  = (new_H - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))

    # 결과 딕셔너리 구성
    res = dict(
        img = ImgNorm(img)[None],
        true_shape = np.int32([img.size[::-1]]),
        unnormalized_img = np.asarray(img),
    )

    if return_transformation:
        scale_w = W1 / new_W
        scale_h = H1 / new_H
        # 크롭 오프셋
        half_crop_w = left
        half_crop_h = top
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res

'''
def resize_img(
        img,
        size,                            # 224·448·512·518
        square_ok=False,
        return_transformation=False
    ):
    assert size in [224, 448, 512, 518, 560]
    MULT = 112                          # 14 and 16 both satisfied

    # 1. numpy [0,1] → PIL
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size

    # 2. resize (224: 짧은 변 맞춤, 그 밖: 긴 변 맞춤)
    if size == 224:
        img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
    else:
        img = _resize_pil_image(img, size)

    W, H = img.size
    cx, cy = W // 2, H // 2

    # 3. 중앙 crop → H,W 모두 112의 배수
    if size == 224:
        half = (min(cx, cy) // MULT) * MULT
        img = img.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw = (cx // MULT) * MULT
        halfh = (cy // MULT) * MULT
        if not square_ok and W == H:
            halfh = int(0.75 * halfw)
            halfh = (halfh // MULT) * MULT
        img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

    res = dict(
        img=ImgNorm(img)[None],                 # (1,3,H,W)
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )

    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
'''

'''
def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
'''