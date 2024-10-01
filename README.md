SCIP-Net
==============================

SCIP-Net for potential outcome prediction in continuous time.

### Setup
Please set up a virtual environment and install the libraries as given in the requirements file.
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## MlFlow
To start an experiments server, run: 

`mlflow server --port=3335`

Connect via ssh to access the MlFlow UI:

`ssh -N -f -L localhost:3335:localhost:3335 <username>@<server-link>`

Then, one can go to the local browser <http://localhost:3335>.

## Experiments

The main training script `config/config.yaml` is run automatically for all models and datasets.
___
The training `<script>` for each different models specified by:

**CRN**: `runnables/train_enc_dec.py`

**CT**: `runnables/train_multi.py`

**RMSNs**: `runnables/train_rmsn.py`

**G-Net**: `runnables/train_gnet.py`

**TE-CDE**: `runnables/train_enc_dec.py`

**SCIP-Net**: `runnables/train_scip.py`

___

The `<backbone>` is specified by:

**CRN**: `crn`

**CT**: `ct`

**RMSNs**: `rmsn`

**G-Net**: `gnet`

**TE-CDE**: `tecde`

**SCIP-Net**: `scip`
___

The `<hyperparameter>` configuration for each model is specified by:

**CRN**: `backbone/crn_hparams='HPARAMS'`

**CT**: `backbone/ct_hparams='HPARAMS'`

**RMSNs**: `backbone/rmsn_hparams='HPARAMS'`

**G-Net**: `backbone/gnet_hparams='HPARAMS'`

**TE-CDE**: `backbone/tecde_hparams='HPARAMS'`

**SCIP-Net**: `backbone/scip_hparams='HPARAMS'`

`HPARAMS` is either one of:
`hparams_grid.yaml` / `hparams_tuned.yaml`.

___

The `<dataset>` is specified by:

**Tumor growth (synthetic)**: `cancer_sim`

**MIMIC-III (semi-synthetic)**: `mimic3_synthetic`
___

Please use the following commands to run the experiments. 
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 <script> +dataset=<dataset> +backbone=<backbone> +<hyperparameter> exp.seed=<seed> exp.logging=True
```

## Example usage
To run our SCIP-Net with optimized hyperparameters on synthetic data with random seeds 101--105 and confounding level 5.0, use the command:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_scip.py --multirun +dataset=cancer_sim +backbone=scip +backbone/scip_hparams='hparams_tuned' dataset.coeff=5.0 exp.seed=101,102,103,104,105
```

To run our SCIP with optimized hyperparameters on semi-synthetic data with random seeds 101--105, use the command:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train_scip.py --multirun +dataset=mimic3_synthetic +backbone=scip +backbone/scip_hparams='hparams_tuned' exp.seed=101,102,103,104,105
```

To run our CIP-ablation, simply turn off the stabilization via:
```console
+exp.stabilize=False
```


Note that, before running semi-synthetic experiments, the MIMIC-III-extract dataset ([all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract)) needs to be placed in `data/processed/`.

___

