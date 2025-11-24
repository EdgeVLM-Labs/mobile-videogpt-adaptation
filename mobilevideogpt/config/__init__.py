import os
from .dataset_config import *

# Build MobileGPT dataset list
_mobilegpt_datasets = [ST_ACADEMIC_MC_0_30_S, ST_ACADEMIC_OE_0_30_S, ST_ACADEMIC_MC_1_2_M, ST_ACADEMIC_OE_1_2_M , ST_ACADEMIC_MC_2_3_M, ST_ACADEMIC_OE_2_3_M, ST_ACADEMIC_MC_30_60_S, ST_ACADEMIC_OE_30_60_S, LONGVU_VIDEO_QA, CLASSIFICATION_K710, CLASSIFICATION_SSV2, CONV_VideoChatGPT, REASONING_NExTQA, REASONING_CLEVRER_QA, REASONING_CLEVRER_MC, VQA_WEBVID_QA, ST_PERCEPTIONTEST_0_30_S, ST_PERCEPTIONTEST_30_60_S, ACTIVITYNET_QA_0_30_S, ACTIVITYNET_QA_30_60_S, ACTIVITYNET_QA_1_2_M, ACTIVITYNET_QA_2_3_M]

# Auto-include QVED if present
if os.path.exists(QVED_TRAIN["annotation_path"]):
    _mobilegpt_datasets.append(QVED_TRAIN)

DataConfig = {
    "PRETRAINING": [CC3M_595K, COCO_CAP, COCO_REG, COCO_REC],
    "MobileGPT": _mobilegpt_datasets,
    "QVED_TRAIN": [QVED_TRAIN],  # QVED training set only
    "QVED_VAL": [QVED_VAL],      # QVED validation set only
    "QVED_TEST": [QVED_TEST],    # QVED test set only
    "QVED": [QVED_TRAIN],        # Backward compatibility - defaults to train
}
