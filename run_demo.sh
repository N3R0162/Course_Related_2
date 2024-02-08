# image, bbox gen
# python lib/boundingbox.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1 snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10
# python lib/boundingbox.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/paketa_24101994_180_session1 snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10
# python lib/boundingbox.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/simon_16101988_196_session2 snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10

# image, WFLW, demo
python lib/demo.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py data/WFLW/images_test_3 snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10


#python lib/demo.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py images/2.jpg
# image, mobilenet_v2
# python lib/demo.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py /home/kyv/Desktop/PIPNet/data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10
# python lib/demo.py experiments/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10.py /home/kyv/Desktop/PIPNet/data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/WFLW/MobilenetV2_2/
# python lib/demo.py experiments/WFLW/pip_32_16_60_mbv3large_l2_l1_10_1_nb10.py /home/kyv/Desktop/PIPNet/data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/WFLW/pip_32_16_60_mbv3large_l2_l1_10_1_nb10

# video 
# python lib/demo_video.py experiments/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10.py videos/002.avi
#python lib/demo_video.py experiments/data_300W_CELEBA/pip_32_16_60_r18_l2_l1_10_1_nb10_wcc.py videos/007.avi

# camera, LaPa, GroundTrust,
# python lib/demo_video.py experiments/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10.py /dev/video2 snapshots/LaPa/GroundTrust
# python lib/demo_video.py experiments/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10.py /dev/video2 snapshots/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10


# camera, WFLW, Mobilenet and Resnet18
# python lib/demo_video.py experiments/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10.py /dev/video2 snapshots/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10
# python lib/demo_video.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py /dev/video2 snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10
# python lib/demo_video.py experiments/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10.py /dev/video2 snapshots/WFLW/MobilenetV2_2/
# python lib/demo_video.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py /dev/video2 snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10
# python source/demo_video.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py /dev/video2 snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10

# Image, WFLW, Mobilenet
# python lib/demo.py experiments/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/WFLW/pip_32_16_60_mbv2_l2_l1_10_1_nb10

#Image, Iris, Resnet18
# python lib/demo_iris.py experiments/Iris/pip_32_16_60_r18_l2_l1_10_1_nb10.py data/Iris/images_train snapshots/Iris/pip_32_16_60_r18_l2_l1_10_1_nb10

#Image, Iris, Resnet18
# python lib/demo.py experiments/Iris/pip_32_16_60_r18_l2_l1_10_1_nb10.py /home/kyv/Desktop/PIPNet/data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/Iris/pip_32_16_60_r18_l2_l1_10_1_nb10

# Image, WFLW, Resnet18
# python lib/demo.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py /home/kyv/Desktop/toKy/Bspline/data/images_val snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10
# python lib/demo.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py /home/kyv/Desktop/toKy/Bspline/data/images_val snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10
# Image, WFLW, Resnet101
# python lib/demo.py experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10
# Image, LaPa, Groundtrust
# python lib/demo.py experiments/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/LaPa/GroundTrust
# Image, LaPa, 18 points
# python lib/demo.py experiments/LaPa/pip_32_16_60_r18_l2_l1_10_1_nb10.py data/WFLW/WFLW_images/inference_images/kalpesh_15061980_171_session1/ snapshots/LaPa/18_points

