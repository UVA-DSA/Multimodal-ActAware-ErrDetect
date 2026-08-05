# Data preparation

The `data/` directory is not distributed with this repository. This page documents how
to obtain each dataset and the exact layout the code expects. On a cluster where the
datasets already exist, a symlink is enough:

```bash
ln -s /path/to/prepared/data ./data
```

## JIGSAWS (dry-lab: Suturing + Needle Passing)

Sources:
* Video and kinematics: [JIGSAWS release](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/)
* Executional error annotations: Hutchinson et al., *Analysis of Executional and
  Procedural Errors in Dry-lab Robotic Surgery Experiments* (labels per gesture segment)
* Context-state (instrument-object interaction) labels: [COMPASS](https://arxiv.org/abs/2209.06424)

Expected layout:

```
data/
├── Suturing/
│   ├── errors/           Suturing_S02_T01.csv ...   # per-gesture rows:
│   │                                                # start_time,end_time,gesture,error1_nor0
│   └── kinematics/       Suturing_S02_T01.csv ...   # 30 Hz PSM kinematics (76 cols; the loader
│                                                    # uses PSML/PSMR position, velocity, gripper angle)
├── Needle_Passing/
│   ├── errors/ ...
│   └── kinematics/ ...
├── vid_frames/
│   ├── Suturing/Suturing_S02_T01/frame_0000.png ... # frames extracted at 10 Hz from the 30 Hz
│   │                                                # video; original frame ids are preserved
│   │                                                # (0, 3, 6, ...)
│   └── Needle_Passing/...
└── vid_features/{resnet50,clip-vit-base-patch16,surgvlp}/
    └── {task}/{video}/features.pt                   # dict: frame_id -> feature tensor
```

Timing: frames on disk are 10 Hz. Training scripts use `--frame_subsample 2` by default,
giving the paper's 5 Hz rate; error annotations address original 30 Hz frame ids, and
kinematics rows are matched to kept frames by those preserved ids (`--kin_downsample 3`).

Generate features after placing frames:

```bash
python jigsaws/extract_features.py --model resnet50
python jigsaws/extract_features.py --model clip-vit-base-patch16
python jigsaws/extract_surgvlp_features_jigsaws.py      # optional, SurgVLP
```

Splits: `splits/LOSO` (leave-one-supertrial-out, folds 1-5) and `splits/LOUO`
(leave-one-user-out, subjects 2-6, 8, 9) contain `train.csv`/`test.csv` per fold and can
be regenerated with `python jigsaws/build_jigsaws_splits.py`.

For the Activity-Aware Visual Embeddings setup, finetuned per-frame features are written to:

```
data/gesture_prompt_features/{ckpt_stem}/{task}/{video}/features.pt
data/error_prompt_features/{ckpt_stem}/{task}/{video}/features.pt
```

## SAR-RARP50 (in-vivo suturing)

Sources:
* Videos + gesture annotations: [SAR-RARP50 challenge](https://rdr.ucl.ac.uk/articles/dataset/SAR-RARP50_train_set/24932529)
* Error annotations + DINOv2 feature pkls: [SEDMamba](https://github.com/wzjialang/SEDMamba)

Expected layout (under `data/SAR_RARP50/`):

```
data/SAR_RARP50/
├── training_set/video_XX/
│   ├── images/000000000.png, 000000012.png, ...     # frames at 5 Hz (every 12th of 60 Hz)
│   └── embed/embed.npy                              # frozen ResNet50 features (make_rarp_embed.py)
│       (optional: ges_embed.npy / tri_embed.npy for the Img+Label setup)
├── testing_set/video_XX/...
├── train_emb_DINOv2/video_XX.pkl                    # {'feature': (T,1000), 'error_GT': (T,), 'image_name'}
├── test_emb_DINOv2/video_XX.pkl                     # frame-level error labels at 5 Hz
├── gestures/{train,test}/video_XX/action_discrete.txt
├── vid_features/{model}/{training_set,testing_set}/{video}/features.pt   # optional extra encoders
└── prompt_features_surgvlp/prompts.pt               # optional SurgVLP prompt embeddings
```

The paper uses 40 training videos and 8 test videos; frame-level `error_GT` from the
pkls provides labels, and windows take the max label over their frames.

Feature generation:

```bash
python SAR_RARP/make_rarp_embed.py                         # ResNet50 embed.npy
python SAR_RARP/extract_features_sar_rarp50.py --model clip-vit-base-patch16   # optional
python SAR_RARP/extract_surgvlp_features_sar_rarp50.py     # optional, SurgVLP
```

Context-state labels for interaction prompts were derived from the challenge's
segmentation masks with the rule-based method described in the paper (Sec. IV-A).
