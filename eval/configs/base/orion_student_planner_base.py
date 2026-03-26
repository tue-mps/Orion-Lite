point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light', 'pedestrian', 'others'
]
dataset_type = 'B2DOrionDataset'
data_root = 'data/bench2drive'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True,
        with_light_state=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='VADObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.37, 0.45),
            final_dim=(320, 640),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=False),
        training=True),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
            'ego_pose_inv', 'command'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
            'ego_lcf_feat', 'can_bus', 'traffic_state_mask', 'traffic_state',
            'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
            'ego_pose_inv', 'command'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='VADObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=dict(
            resize_lim=(0.37, 0.45),
            final_dim=(320, 640),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=False),
        training=False),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
                    'ego_pose_inv', 'command'
                ],
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'can_bus', 'fut_valid_flag',
                    'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
                    'ego_pose_inv', 'command'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type='B2DOrionDataset',
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True,
                with_light_state=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=True),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='PETRFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                collect_keys=[
                    'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
                    'ego_pose_inv', 'command'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'can_bus',
                    'traffic_state_mask', 'traffic_state', 'lidar2img',
                    'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv',
                    'command'
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        seq_mode=True,
        seq_split_num=1,
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        queue_length=1,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        polyline_points_num=11),
    val=dict(
        type='B2DOrionDataset',
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=False),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'cam_intrinsic', 'timestamp',
                            'ego_pose', 'ego_pose_inv', 'command'
                        ],
                        class_names=[
                            'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                            'traffic_cone', 'traffic_light', 'pedestrian',
                            'others'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'ego_his_trajs', 'gt_attr_labels', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'can_bus', 'fut_valid_flag', 'lidar2img',
                            'cam_intrinsic', 'timestamp', 'ego_pose',
                            'ego_pose_inv', 'command'
                        ])
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        queue_length=1,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        polyline_points_num=11,
        eval_cfg=dict(
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            mean_ap_weight=5,
            class_names=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light', 'pedestrian'
            ],
            tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            err_name_maping=dict(
                trans_err='mATE',
                scale_err='mASE',
                orient_err='mAOE',
                vel_err='mAVE',
                attr_err='mAAE'),
            class_range=dict(
                car=(50, 50),
                van=(50, 50),
                truck=(50, 50),
                bicycle=(40, 40),
                traffic_sign=(30, 30),
                traffic_cone=(30, 30),
                traffic_light=(30, 30),
                pedestrian=(40, 40)))),
    test=dict(
        type='B2DOrionDataset',
        data_root='data/bench2drive',
        ann_file='data/infos/b2d_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='VADObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='VADObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='ResizeCropFlipRotImage',
                data_aug_conf=dict(
                    resize_lim=(0.37, 0.45),
                    final_dim=(320, 640),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    H=900,
                    W=1600,
                    rand_flip=False),
                training=False),
            dict(
                type='ResizeMultiview3D',
                img_scale=(640, 640),
                keep_ratio=False,
                multiscale_mode='value'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'cam_intrinsic', 'timestamp',
                            'ego_pose', 'ego_pose_inv', 'command'
                        ],
                        class_names=[
                            'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                            'traffic_cone', 'traffic_light', 'pedestrian',
                            'others'
                        ],
                        with_label=False),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'ego_his_trajs', 'gt_attr_labels', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'can_bus', 'fut_valid_flag', 'lidar2img',
                            'cam_intrinsic', 'timestamp', 'ego_pose',
                            'ego_pose_inv', 'command'
                        ])
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_root='data/bench2drive/maps',
        map_file='data/infos/b2d_map_infos.pkl',
        queue_length=1,
        past_frames=2,
        future_frames=6,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        polyline_points_num=11,
        eval_cfg=dict(
            dist_ths=[0.5, 1.0, 2.0, 4.0],
            dist_th_tp=2.0,
            min_recall=0.1,
            min_precision=0.1,
            mean_ap_weight=5,
            class_names=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light', 'pedestrian'
            ],
            tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
            err_name_maping=dict(
                trans_err='mATE',
                scale_err='mASE',
                orient_err='mAOE',
                vel_err='mAVE',
                attr_err='mAAE'),
            class_range=dict(
                car=(50, 50),
                van=(50, 50),
                truck=(50, 50),
                bicycle=(40, 40),
                traffic_sign=(30, 30),
                traffic_cone=(30, 30),
                traffic_light=(30, 30),
                pedestrian=(40, 40)))),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=10,
        warmup_split_num=80,
        num_iters_to_seq=917),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=6419,
    pipeline=[
        dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=True),
        dict(
            type='VADObjectRangeFilter',
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        dict(
            type='VADObjectNameFilter',
            classes=[
                'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                'traffic_cone', 'traffic_light', 'pedestrian', 'others'
            ]),
        dict(
            type='ResizeCropFlipRotImage',
            data_aug_conf=dict(
                resize_lim=(0.37, 0.45),
                final_dim=(320, 640),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                H=900,
                W=1600,
                rand_flip=False),
            training=False),
        dict(
            type='ResizeMultiview3D',
            img_scale=(640, 640),
            keep_ratio=False,
            multiscale_mode='value'),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='PadMultiViewImage', size_divisor=32),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PETRFormatBundle3D',
                    collect_keys=[
                        'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose',
                        'ego_pose_inv', 'command'
                    ],
                    class_names=[
                        'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                        'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                    ],
                    with_label=False),
                dict(
                    type='CustomCollect3D',
                    keys=[
                        'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                        'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                        'ego_fut_cmd', 'ego_lcf_feat', 'can_bus',
                        'fut_valid_flag', 'lidar2img', 'cam_intrinsic',
                        'timestamp', 'ego_pose', 'ego_pose_inv', 'command'
                    ])
            ])
    ])
checkpoint_config = dict(interval=917, max_keep_ckpts=3)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'adzoo/orion/work_dirs/exp9_two_stage_20260317_152208/exp9_stage2'
load_from = 'adzoo/orion/work_dirs/exp9_two_stage_20260317_152208/exp9_stage1/latest.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
map_classes = [
    'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign'
]
queue_length = 1
map_fixed_ptsnum_per_gt_line = 11
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = 6
past_frames = 2
future_frames = 6
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
ida_aug_conf = dict(
    resize_lim=(0.37, 0.45),
    final_dim=(320, 640),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=False)
occflow_grid_conf = dict(
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0])
NameMapping = dict({
    'vehicle.bh.crossbike':
    'bicycle',
    'vehicle.diamondback.century':
    'bicycle',
    'vehicle.gazelle.omafiets':
    'bicycle',
    'vehicle.audi.etron':
    'car',
    'vehicle.chevrolet.impala':
    'car',
    'vehicle.dodge.charger_2020':
    'car',
    'vehicle.dodge.charger_police':
    'car',
    'vehicle.dodge.charger_police_2020':
    'car',
    'vehicle.lincoln.mkz_2017':
    'car',
    'vehicle.lincoln.mkz_2020':
    'car',
    'vehicle.mini.cooper_s_2021':
    'car',
    'vehicle.mercedes.coupe_2020':
    'car',
    'vehicle.ford.mustang':
    'car',
    'vehicle.nissan.patrol_2021':
    'car',
    'vehicle.audi.tt':
    'car',
    'vehicle.ford.crown':
    'car',
    'vehicle.tesla.model3':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
    'van',
    'vehicle.ford.ambulance':
    'van',
    'vehicle.carlamotors.firetruck':
    'truck',
    'traffic.speed_limit.30':
    'traffic_sign',
    'traffic.speed_limit.40':
    'traffic_sign',
    'traffic.speed_limit.50':
    'traffic_sign',
    'traffic.speed_limit.60':
    'traffic_sign',
    'traffic.speed_limit.90':
    'traffic_sign',
    'traffic.speed_limit.120':
    'traffic_sign',
    'traffic.stop':
    'traffic_sign',
    'traffic.yield':
    'traffic_sign',
    'traffic.traffic_light':
    'traffic_light',
    'static.prop.warningconstruction':
    'traffic_cone',
    'static.prop.warningaccident':
    'traffic_cone',
    'static.prop.trafficwarning':
    'traffic_cone',
    'static.prop.constructioncone':
    'traffic_cone',
    'walker.pedestrian.0001':
    'pedestrian',
    'walker.pedestrian.0003':
    'pedestrian',
    'walker.pedestrian.0004':
    'pedestrian',
    'walker.pedestrian.0005':
    'pedestrian',
    'walker.pedestrian.0007':
    'pedestrian',
    'walker.pedestrian.0010':
    'pedestrian',
    'walker.pedestrian.0013':
    'pedestrian',
    'walker.pedestrian.0014':
    'pedestrian',
    'walker.pedestrian.0015':
    'pedestrian',
    'walker.pedestrian.0016':
    'pedestrian',
    'walker.pedestrian.0017':
    'pedestrian',
    'walker.pedestrian.0018':
    'pedestrian',
    'walker.pedestrian.0019':
    'pedestrian',
    'walker.pedestrian.0020':
    'pedestrian',
    'walker.pedestrian.0021':
    'pedestrian',
    'walker.pedestrian.0022':
    'pedestrian',
    'walker.pedestrian.0025':
    'pedestrian',
    'walker.pedestrian.0027':
    'pedestrian',
    'walker.pedestrian.0030':
    'pedestrian',
    'walker.pedestrian.0031':
    'pedestrian',
    'walker.pedestrian.0032':
    'pedestrian',
    'walker.pedestrian.0034':
    'pedestrian',
    'walker.pedestrian.0035':
    'pedestrian',
    'walker.pedestrian.0041':
    'pedestrian',
    'walker.pedestrian.0042':
    'pedestrian',
    'walker.pedestrian.0046':
    'pedestrian',
    'walker.pedestrian.0047':
    'pedestrian',
    'static.prop.dirtdebris01':
    'others',
    'static.prop.dirtdebris02':
    'others'
})
eval_cfg = dict(
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    min_recall=0.1,
    min_precision=0.1,
    mean_ap_weight=5,
    class_names=[
        'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
        'traffic_light', 'pedestrian'
    ],
    tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    err_name_maping=dict(
        trans_err='mATE',
        scale_err='mASE',
        orient_err='mAOE',
        vel_err='mAVE',
        attr_err='mAAE'),
    class_range=dict(
        car=(50, 50),
        van=(50, 50),
        truck=(50, 50),
        bicycle=(40, 40),
        traffic_sign=(30, 30),
        traffic_cone=(30, 30),
        traffic_light=(30, 30),
        pedestrian=(40, 40)))
use_memory = True
num_gpus = 8
batch_size = 32
num_iters_per_epoch = 917
num_epochs = 6
llm_path = 'ckpts/pretrain_qformer/'
use_gen_token = True
use_col_loss = True
collect_keys = [
    'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv',
    'command'
]
model = dict(
    type='OrionStudentPlanner',
    save_path='./results_planning_only/',
    use_grid_mask=True,
    frozen=False,
    use_lora=False,
    tokenizer=None,
    lm_head=None,
    use_gen_token=False,
    use_diff_decoder=False,
    use_col_loss=True,
    loss_plan_reg=dict(type='L1Loss', loss_weight=3.0),
    loss_plan_bound=dict(
        type='PlanMapBoundLoss', loss_weight=3.0, dis_thresh=1.0),
    loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=1.0),
    loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=3.0),
    img_backbone=dict(
        type='EVAViT',
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.6666666666666665,
        window_block_indexes=[
            0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22
        ],
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True,
        frozen=False),
    map_head=dict(
        type='OrionHeadM',
        num_classes=6,
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        code_weights=[1.0, 1.0],
        score_threshold=0.2,
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='LaneHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.02),
                iou_cost=dict(type='IoUCost', weight=0.0))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.02),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)),
    pts_bbox_head=dict(
        type='OrionHead',
        num_classes=9,
        in_channels=1024,
        out_dims=4096,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,
        n_control=11,
        match_with_velo=False,
        pred_traffic_light_state=True,
        use_col_loss=True,
        use_memory=True,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        use_pe=False,
        motion_transformer_decoder=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=512,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            compensate=[0, 0, 0.3, 0, 0, 0, 0, 0.3, 0],
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1),
        memory_decoder_transformer=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=512,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_traffic=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))),
    student_cfg=dict(
        input_dim=4096,
        hidden_dim=1024,
        output_dim=4096,
        num_layers=6,
        num_heads=16,
        dropout=0.1),
    img_backbone_ckpt='ckpts/eva02_petr_proj.pth',
    img_backbone_prefix='img_backbone.',
    orion_ckpt_load_mode='eva_vision_orion_rest',
    orion_ckpt='/data/video-fm/datasets/jgu/Orion/ckpts/Orion.pth')
info_root = 'data/infos'
map_root = 'data/bench2drive/maps'
map_file = 'data/infos/b2d_map_infos.pkl'
ann_file_train = 'data/infos/b2d_infos_train.pkl'
ann_file_val = 'data/infos/b2d_infos_val.pkl'
ann_file_test = 'data/infos/b2d_infos_val.pkl'
inference_only_pipeline = [
    dict(
        type='LoadMultiViewImageFromFilesInCeph',
        to_float32=True,
        file_client_args=dict(backend='disk'),
        img_root='data/bench2drive'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=['img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'command'])
        ])
]
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=4e-05,
    betas=(0.9, 0.999),
    weight_decay=1e-05,
    paramwise_cfg=dict(
        decay_rate=0.9,
        head_decay_rate=1.0,
        lm_head_decay_rate=0.1,
        decay_type='vit_wise',
        num_layers=24))
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
find_unused_parameters = False
runner = dict(type='IterBasedRunner', max_iters=5502)
custom_imports = dict(
    imports=[
        'experiments.student_orion.orion_student_planner',
        'experiments.student_orion.vision_freeze_hook'
    ],
    allow_failed_imports=False)
custom_hooks = [
    dict(
        type='StudentBackboneLoadDebugHook',
        fail_on_violation=True,
        run_once=True)
]
gpu_ids = range(0, 8)
