# SPLIT=train python src/preprocess_InterHand26M_mp.py
# SPLIT=val python src/preprocess_InterHand26M_mp.py
# SPLIT=test python src/preprocess_InterHand26M_mp.py

# SRC=ih26m_train_wds_output/ih26m_train-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train2/%06d.tar\
#     python src/data_reorganizer.py
# SRC=ih26m_test_wds_output/ih26m_test-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test2/%06d.tar\
#     python src/data_reorganizer.py
# SRC=ih26m_val_wds_output/ih26m_val-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/val2/%06d.tar\
#     python src/data_reorganizer.py

# SETUP=s1 SPLIT=val python src/preprocess_DexYCB_mp.py
# SETUP=s1 SPLIT=test python src/preprocess_DexYCB_mp.py
# SETUP=s1 SPLIT=train python src/preprocess_DexYCB_mp.py

# SRC=dexycb_s1_train_wds_output/dexycb_s1_train-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/train/%06d.tar\
#     python src/data_reorganizer.py
# SRC=dexycb_s1_test_wds_output/dexycb_s1_test-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/test/%06d.tar\
#     python src/data_reorganizer.py
# SRC=dexycb_s1_val_wds_output/dexycb_s1_val-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/DexYCB/s1/val/%06d.tar\
#     python src/data_reorganizer.py

SPLIT=train python src/preprocess_HO3D_mp.py
SRC=ho3d_train_wds_output/ho3d_train-worker*.tar \
DST=/mnt/qnap/data/datasets/webdatasets/HO3D_v3/train/%06d.tar\
    python src/data_reorganizer.py
# SPLIT=evaluation python src/preprocess_HO3D_mp.py
# SRC=ho3d_evaluation_wds_output/ho3d_evaluation-worker*.tar \
# DST=/mnt/qnap/data/datasets/webdatasets/HO3D_v3/evaluation/%06d.tar\
#     python src/data_reorganizer.py