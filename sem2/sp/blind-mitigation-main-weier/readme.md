# blind-mitigation
## Repository's structure
```
├── data
│   ├── beerslaw
│   └── PISA
├── notebooks
│   ├── 0_data_processing
│   ├── 1_algos_investigation
│   └── 2_dirty_prototypes
│   	├── oversampling
│   	└── clustering
├── src
│   ├── utils
│   ├── configs
│   ├── data_handlers
│   ├── ml
│   └── visualisers
├── experiments
│   ├── oversample
│   └── clustering
└── reports
    ├── presentation
    └── reports
```

## Launching the oversampling script
1. configure the experiment by editing the `src/configs/oversampling_config.yaml`
2. place yourself in the `src` folder
3. run `python script_oversampling --nonnested`

