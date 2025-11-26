# SPLIT=train python src/preprocess_InterHand26M_mp.py
# SPLIT=val python src/preprocess_InterHand26M_mp.py
# SPLIT=test python src/preprocess_InterHand26M_mp.py

SRC=ih26m_train_wds_output/ih26m_train-worker*.tar \
DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/train2/%06d.tar\
    python src/data_reorganizer.py
SRC=ih26m_test_wds_output/ih26m_test-worker*.tar \
DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/test2/%06d.tar\
    python src/data_reorganizer.py
SRC=ih26m_val_wds_output/ih26m_val-worker*.tar \
DST=/mnt/qnap/data/datasets/webdatasets/InterHand2.6M/val2/%06d.tar\
    python src/data_reorganizer.py