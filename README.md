# scMISP:scMISP: a subspace projection-based multi-scale information fusion framework for
single-cell multi-omics clustering

![framework](images/SpaMICS.png)

## Requirement
- torch==1.11.0
- python==3.7.12
- numpy==1.18.5
- pandas==1.3.5
- scikit-learn==1.0.2
- scanpy==1.9.3
- scipy==1.4.1
- anndata==0.8.0
- h5py==3.8.0

## Usage
#### Clone this repo.
```
git clone https://github.com/Oyl-CityU/scMISP.git
```

#### Example command
Take the dataset pbmccite" as an example
```
python main.py --dataset "pbmccite"
```