## Requirements
To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```


## Dataset

    .data/wikievents/
	├── train.jsonl        # 206 docs 
	├── dev.jsonl          # 20  docs
	└──  test.jsonl         # 20  docs
	
    .data/rams/
	├── train.jsonlines    # 7329 docs 
	├── dev.jsonlines      # 924 docs
	└── test.jsonlines     # 871 docs

## Running

Run on WikiEvents with:
```
sh run_wikievents_setting3.sh
```

Run on RAMS with:
```
sh run_rams_setting3.sh 
```

Model checkpoints and logs will be saved to `./saved_models/.


