# yamyam-lab

This repository aims for developing recommender system using review data in kakao [map](https://map.kakao.com/).

## Environment setting

We use [poetry](https://github.com/python-poetry/poetry) to manage dependencies of repository.

It is recommended that latest version of poetry should be installed in advance.

```shell
$ poetry --version
Poetry (version 1.8.5)
```

Python version should be higher than `3.11`.

```shell
$ python --version
Python 3.11.11
```

If python version is lower than `3.11`, try installing required version using `pyenv`.

Create virtual environment.

```shell
$ poetry shell
```

After setting up python version, just run following command which will install all the required packages from `poetry.lock`.

```shell
$ poetry install
```

### Note

If you want to add package to `pyproject.toml`, please use following command.

```shell
$ poetry add "package==1.0.0"
```

Then, update `poetry.lock` to ensure that repository members share same environment setting.

```shell
$ poetry lock
```

## Experiment results

|Algorithm|Task|mAP@3|mAP@7|mAP@10|NDCG@3|NDCG@7|NDCG@10|
|----------------|---------|------|------|------|-------|-------|-------|
|SVD|Regression|0.00087|0.00066|0.00065|0.00127|0.00148|0.00183|
|node2vec|Embedding|0.0058|0.00384|0.00355|0.00876|0.00918|0.01006|