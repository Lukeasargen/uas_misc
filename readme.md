
```
conda create --name uas python=3.9 -y
conda activate uas
pip install -r requirements.txt
conda install ipykernel jupyter -y
python -m ipykernel install --user --name uas --display-name "uas"
```
