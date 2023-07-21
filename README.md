# SC_model

SC Model is an inorganic material crystal materials synthesizability score prediction model


## Installation

Please use Python 3.7.7 to run the model.

To install, just clone the repository. Then install all required packages:

```bash
pip install -r requirement.txt
```

## Usage

You can train and test the ternary SC model by:

```bash
python main.py
```

You can choose to include or exclude FTCP feature set, or use only atomic features by running:

```bash
python main.py --rp include_ftcp
python main.py --rp exclude_ftcp
python main.py --rp atomic
```
You can also use pre-trained compositional model or full crystal representation model to predict SC for compounds:

```bash
python .\predict.py --type formula --crystal CaTiO3
python .\predict.py --type cif --crystal .\data\predict_cif\CaTiO3_mp-4019_conventional_standard.cif
```

   

## Authors
The code was primarily written by Ruiming Zhu, under supervision of Prof. Kedar Hippalgaonkar.
