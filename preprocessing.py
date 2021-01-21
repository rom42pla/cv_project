import random
import time
import os
import json

from modules.video_preprocessing import *
from modules.features_extraction import FeatureExtractor
from modules.HDF5Store import *

random.seed(69420)
# np.random.seed(69420)


if __name__ == "__main__":
    DATA_PATH = "assets/datasets/"
    VIDEO_PATH = DATA_PATH + "video/"
    SAMPLES_PATH = DATA_PATH + "samples/"
    DIM = 128
    WSIZE, WSLIDE, KEEP = 30, 5, 8  # diminuire WSIZE (?)
    CHANNELS = 3
    QUEUE_SZ = 512
    BATCH_SZ = QUEUE_SZ // 8
    # TRAIN_SPLIT = 0.85
    assert BATCH_SZ <= QUEUE_SZ

    # with open("assets/model/vocab.json", "r") as f:
    with open("assets/model/vocab_groups.json", "r") as f:
        TARGETS_VOCAB = json.load(f)

    start_time = time.time()

    fstore_train = SAMPLES_PATH + "train_%dx%d_rgb-canny-lk_wsize%d_wslide%d.h5" % (DIM, DIM, WSIZE, WSLIDE)
    fstore_valid = SAMPLES_PATH + "valid_%dx%d_rgb-canny-lk_wsize%d_wslide%d.h5" % (DIM, DIM, WSIZE, WSLIDE)

    dss = [("1_rgb", (KEEP, CHANNELS, DIM, DIM), np.float32),
           ("2_canny", (KEEP, 1, DIM, DIM), np.float32),
           ("3_lk", (KEEP - 1, 2, DIM, DIM), np.float32),
           ("4_targets", (), np.int32)]

    if os.path.isfile(fstore_train): os.remove(fstore_train)
    if os.path.isfile(fstore_valid): os.remove(fstore_valid)

    store_train = HDF5Store(fstore_train, mode="a")
    store_valid = HDF5Store(fstore_valid, mode="a")

    for dsname, dsshape, dstype in dss:
        store_train.create_dataset(dsname, dsshape, dtype=dstype, chunk_len=32)
        store_valid.create_dataset(dsname, dsshape, dtype=dstype, chunk_len=32)

    fvids = sorted([VIDEO_PATH + f for f in os.listdir(VIDEO_PATH) if
                    f.endswith(".mp4") and os.path.isfile(VIDEO_PATH + f.replace(".mp4", ".txt"))])#[:4]

    random.shuffle(fvids)

    fvids_train, fvids_valid = fvids[:-2], fvids[-2:]

    sm_train = SamplingManager(files=fvids_train, qsz=QUEUE_SZ, batch_sz=BATCH_SZ, dim=DIM, wsize=WSIZE, wslide=WSLIDE,
                               nf_keep=KEEP, store=store_train, feature_extractor=FeatureExtractor,
                               target_groups=TARGETS_VOCAB)

    sm_valid = SamplingManager(files=fvids_valid, qsz=QUEUE_SZ, batch_sz=BATCH_SZ, dim=DIM, wsize=WSIZE, wslide=WSLIDE,
                               nf_keep=KEEP, store=store_valid, feature_extractor=FeatureExtractor,
                               target_groups=TARGETS_VOCAB)

    for t in [sm_train, sm_valid]: t.start()
    for t in [sm_train, sm_valid]: t.join()

    print(time.time() - start_time)

# DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE
# DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE
# DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE # DO NOT DELETE

# dsnames = [ds[0] for ds in dss]

# conv_queue = Queue(maxsize=QUEUE_SZ)

# conv_threads = []
# for fvid in fvids:
#     conv_thread = VideoSampler(VIDEO_PATH + fvid, dim=DIM, wsize=WSIZE, wslide=WSLIDE, queue=conv_queue,
#                                nf_keep=KEEP, feature_extractor=FeatureExtractor, target_groups=TARGETS_VOCAB)
#     conv_thread.start()
#     conv_threads.append(conv_thread)
#
# batch_train, batch_valid = [], []
# batch_sz, ibatch = BATCH_SZ, 0
# while any([th.is_alive() for th in conv_threads]) or not conv_queue.empty():
#     try:
#         sample = conv_queue.get(block=True, timeout=0.5)
#         if random.uniform(0, 1) < TRAIN_SPLIT:
#             batch_train.append(sample)
#         else:
#             batch_valid.append(sample)
#         if len(batch_train) == batch_sz:
#             store_train.append_batch(batch_train, dsnames)
#             batch_train = []
#             store_train.flush()
#         if len(batch_valid) == batch_sz:
#             store_valid.append_batch(batch_valid, dsnames)
#             batch_valid = []
#             store_valid.flush()
#     except _queue.Empty:
#         pass

# if len(batch_train) > 0:
#     store_train.append_batch(batch_train, dsnames)
#     store_train.flush()
# if len(batch_valid) > 0:
#     store_valid.append_batch(batch_valid, dsnames)
#     store_valid.flush()

# print()
# for t in conv_threads: t.join()

# store_train.close()
# store_valid.close()
