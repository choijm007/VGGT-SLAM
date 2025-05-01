import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import time
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_cv2

def relocalization(frame, keyframes, factor_graph, retrieval_database):
    """
    현재 프레임을 사용해 카메라 위치를 재조정(relocalize)하는 함수입니다.
    
    1. retrieval_database를 사용해 현재 프레임과 유사한 키프레임들을 검색합니다.
    2. 유사한 키프레임이 있으면 일시적으로 현재 프레임을 키프레임으로 추가합니다.
    3. factor_graph를 통해 이 프레임과 기존 키프레임 사이의 관계를 계산합니다.
    4. 관계가 충분히 강하면(매칭이 성공하면) 현재 프레임을 데이터베이스에 추가하고 위치를 업데이트합니다.
    5. 매칭이 실패하면 키프레임에서 제거합니다.
    6. 성공적인 재위치화 후에는 전체 그래프를 최적화합니다.
    
    Args:
        frame: 현재 프레임
        keyframes: 키프레임 모음
        factor_graph: 키프레임 간의 관계를 나타내는 그래프
        retrieval_database: 키프레임 검색 데이터베이스
    
    Returns:
        bool: 재위치화 성공 여부
    """
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


#def run_backend(cfg, model, states, keyframes, K):
def run_backend(states, keyframes):
    """
    SLAM 시스템의 백엔드 프로세스를 실행하는 함수입니다.
    
    1. 팩터 그래프와 검색 데이터베이스를 초기화합니다.
    2. 시스템의 모드(초기화, 트래킹, 재위치화 등)에 따라 다른 처리를 수행합니다.
    3. 재위치화 요청이 있으면 relocalization 함수를 호출합니다.
    4. 키프레임이 추가되면 해당 키프레임과 기존 키프레임 간의 관계를 계산합니다.
    5. 루프 클로저 감지 및 처리를 수행합니다.
    6. 그래프 최적화를 수행하여 카메라 위치를 보정합니다.
    
    Args:
        cfg: 설정 정보
        model: MAST3R 신경망 모델
        states: 공유 상태 정보
        keyframes: 키프레임 모음
        K: 카메라 내부 파라미터(intrinsics)
    """
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return
    if mode == Mode.RELOC:
        frame = states.get_frame()
        success = relocalization(frame, keyframes, factor_graph, retrieval_database)
        if success:
            states.set_mode(Mode.TRACKING)
        states.dequeue_reloc()
        return
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )
    """
    retrieval DB를 통해 유사한 오래된 키프레임들을 검색

    add_after_query=True: 검색 끝난 후 해당 프레임을 DB에 추가

    retrieval_inds: loop closure 후보 키프레임 인덱스들

    kf_idx에 합쳐서 정합 대상으로 확장
    """
        
    kf_idx += retrieval_inds

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
    frame_idx = [idx] * len(kf_idx)
    if kf_idx:
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )

    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    if config["use_calib"]:
        factor_graph.solve_GN_calib()
    else:
        factor_graph.solve_GN_rays()

    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    """
    MAST3R SLAM 시스템의 메인 함수입니다.
    
    1. 멀티프로세싱 설정 및 GPU 설정을 초기화합니다.
    2. 명령줄 인자를 파싱하여 설정과 데이터셋을 로드합니다.
    3. 시각화, 백엔드 처리를 위한 멀티프로세스를 시작합니다.
    4. 데이터셋의 각 프레임에 대해:
       - 적절한 모드(초기화, 트래킹, 재위치화)에서 처리합니다.
       - 필요시 키프레임을 추가하고 글로벌 최적화를 요청합니다.
    5. 완료 후 결과(궤적, 3D 재구성, 키프레임)를 저장합니다.
    
    주요 모드:
    - INIT: 시스템 초기화, 첫 번째 키프레임 설정
    - TRACKING: 현재 프레임의 카메라 위치 추적
    - RELOC: 추적 실패 시 재위치화 시도
    """
    mp.set_start_method("spawn") 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager() # 메모리 공유 관리자 생성
    main2viz = new_queue(manager, args.no_viz) # 메인 프로세스에서 시각화 프로세스로 메시지 전달
    viz2main = new_queue(manager, args.no_viz) # 시각화 프로세스에서 메인 프로세스로 메시지 전달   

    dataset = load_dataset(args.dataset) # 데이터셋 로드
    dataset.subsample(config["dataset"]["subsample"]) # 하위 샘플링
    h, w = dataset.get_img_shape()[0] # 이미지 크기

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader) # 캘리브레이션 파일 로드
        config["use_calib"] = True # 캘리브레이션 사용
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w) # 키프레임 공유 메모리 객체
    states = SharedStates(manager, h, w) # 상태 공유 메모리 객체

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device) # 모델 로드
    model.share_memory() # 모델 공유 메모리 할당
    
    # VGGT 모델 로드
    # vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()
    
    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i] # 프레임 로드
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        # 초기 프레임이면 단위 행렬, 그렇지 않으면 마지막 프레임의 행렬
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        # if mode == Mode.INIT:
        #     # Initialize via mono inference, and encoded features neeed for database
        #     X_init, C_init = mast3r_inference_mono(model, frame) # 한장의 이미지로부터 초기 3D 클라우드 + 신뢰도를 추론
        #     frame.update_pointmap(X_init, C_init) # 해당 프레임에 3D 클라우드 + 신뢰도 업데이트
        #     keyframes.append(frame)
        #     states.queue_global_optimization(len(keyframes) - 1)
        #     states.set_mode(Mode.TRACKING)
        #     states.set_frame(frame)
        #     i += 1
        #     continue
        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = tracker.track_init(frame) # 한장의 이미지로부터 초기 3D 클라우드 + 신뢰도를 추론
            frame.update_pointmap(X_init, C_init) # 해당 프레임에 3D 클라우드 + 신뢰도 업데이트
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue
        
        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track_vggt(frame, ) # 프레임 추적
            # 여기서 New Keyframe 추가 여부, 매칭 정보, 재위치화 여부 결정
            """
            현재 프레임과 마지막 키프레임 간 3D 매칭 수행
            현재 카메라 위치 (T_WC) 추정
            키프레임에 새 포인트맵 업데이트
            새로운 키프레임 추가할지 결정
            필요시 재위치화 요청 (try_reloc=True)
            """
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = tracker.track_init(frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()

        # elif mode == Mode.RELOC:
        #     X, C = mast3r_inference_mono(model, frame)
        #     frame.update_pointmap(X, C)
        #     states.set_frame(frame)
        #     states.queue_reloc()
        #     # In single threaded mode, make sure relocalization happen for every frame
        #     while config["single_thread"]:
        #         with states.lock:
        #             if states.reloc_sem.value == 0:
        #                 break
        #         time.sleep(0.01)
        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)

        run_backend(states, keyframes)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    #backend.join()
    if not args.no_viz:
        viz.join()
