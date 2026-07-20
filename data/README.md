# Dataset Setup

Download the public [15-Scene Dataset](https://www.kaggle.com/datasets/zaiyankhan/15scene-dataset) and extract it using one folder per class:

```text
data/15_scene/
├── Bedroom/
├── Coast/
├── Forest/
├── Highway/
├── Industrial/
├── Inside_city/
├── Kitchen/
├── Living_room/
├── Mountain/
├── Office/
├── Open_country/
├── Store/
├── Street/
├── Suburb/
└── Tall_building/
```

The loader sorts class folders lexicographically and assigns integer labels automatically. Raw images are intentionally excluded from this repository; follow the source dataset's license and citation terms.

