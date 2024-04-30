model-name-or-path:=None
dataset-name-or-path:=None

index:
	PYTHONPATH=${PWD} python llma/api index --model-name-or-path=$(model-name-or-path) --dataset-name-or-path=$(dataset-name-or-path)
