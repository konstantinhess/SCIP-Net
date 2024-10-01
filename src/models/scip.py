from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
from torchdiffeq import odeint
from typing import Union
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np


from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils import clip_normalize_stabilized_weights
from src.models.utils_cde import NeuralCDE


logger = logging.getLogger(__name__)


class SCIP(TimeVaryingCausalModel):
    """
    Pytorch-Lightning implementation of Stabilized continuous time inverse propensity weighted network (SCIP-Net)
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'encoder', 'decoder', 'propensity_treatment', 'propensity_history'}
    tuning_criterion = None

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        """
        Args:
            args: DictConfig of model hyperparameters
            dataset_collection: Dataset collection
            autoregressive: Flag of including previous outcomes to modelling
            has_vitals: Flag of vitals in dataset
            bce_weights: Re-weight BCE if used
            **kwargs: Other arguments
        """
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

    def _init_specific(self, sub_args: DictConfig, encoder_r_size: int = None):
        # Encoder/decoder-specific parameters
        try:
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.inhomogeneous = sub_args.inhomogeneous

            # Pytorch model init
            if self.seq_hidden_units is None or self.dropout_rate is None:
                raise MissingMandatoryValue()

            if self.model_type == 'decoder':
                self.memory_adapter = nn.Linear(encoder_r_size, self.seq_hidden_units)

            # self.lstm = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)
            self.cde = NeuralCDE(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate,
                                 inhomogeneous=self.inhomogeneous)

            self.output_layer = nn.Linear(self.seq_hidden_units, self.output_size)

            if self.model_type in ['propensity_treatment', 'propensity_history']:
                self.intensity_layer = nn.Linear(self.seq_hidden_units, 1)

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']
        sub_args.max_grad_norm = new_args['max_grad_norm']

    def get_propensity_scores(self, dataset: Dataset) -> np.array:
        logger.info(f'Propensity scores for {dataset.subset_name}.')
        if self.model_type == 'propensity_treatment' or self.model_type == 'propensity_history':
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
            propensity_scores = torch.cat([x[0] for x in self.trainer.predict(self, data_loader)])
        else:
            raise NotImplementedError()
        return propensity_scores.numpy()

    def get_intensity_scores(self, dataset: Dataset) -> np.array:
        logger.info(f'Intensity scores for {dataset.subset_name}.')
        if self.model_type == 'propensity_treatment' or self.model_type == 'propensity_history':
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
            intensity_scores = torch.cat([x[1] for x in self.trainer.predict(self, data_loader)])
        else:
            raise NotImplementedError()
        return intensity_scores.numpy()

class SCIPPropensityNetworkTreatment(SCIP):

    model_type = 'propensity_treatment'
    tuning_criterion = 'bce'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + 1
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_treatments

        self._init_specific(args.model.propensity_treatment)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            assert self.hparams.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch):
        prev_treatments = batch['prev_treatments']
        prev_treatment_times = batch['prev_treatment_times']
        x = torch.cat((prev_treatments, prev_treatment_times), dim=-1)
        x = self.cde(x, init_states=None, device=self.device)
        propensity_pred = self.output_layer(x)
        intensity_pred = self.intensity_layer(x)

        return propensity_pred, intensity_pred

    def training_step(self, batch, batch_ind):
        propensity_pred, intensity_pred = self(batch)

        propensity_loss = self.bce_loss(propensity_pred, batch['current_treatments'].double(), kind='predict')
        propensity_loss = ((batch['active_entries'].squeeze(-1) * batch['current_treatment_times'].squeeze(-1) * propensity_loss).sum() /
                           (batch['active_entries']*batch['current_treatment_times']).sum())

        intensity_loss = self.bce_loss(intensity_pred, batch['current_treatment_times'].double(), kind='predict')
        intensity_loss = (batch['active_entries'].squeeze(-1) * intensity_loss).sum() / batch['active_entries'].sum()

        self.log(f'{self.model_type}_propensity_loss', propensity_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_intensity_loss', propensity_loss, on_epoch=True, on_step=False, sync_dist=True)
        return propensity_loss + intensity_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        propensity_pred, intensity_pred = self(batch)
        prev_obs = torch.cat((batch['prev_treatments'], batch['prev_treatment_times']), dim=-1)

        def intensity_integrand(t, x):
            Z_t = self.cde(prev_obs, init_states=None, t_max=t, device=self.device)[:, -1, :].to(self.device).double()
            return F.sigmoid(self.intensity_layer(Z_t))

        intensity_init = F.sigmoid(self.intensity_layer(self.cde.input_layer(prev_obs[:, 0, :])))
        integrated_intensity = odeint(func=intensity_integrand, y0=intensity_init, method='euler',
                                      t=torch.linspace(0, 1, prev_obs.size(1), device=self.device)).permute(1,0,2)

        for i in reversed(range(1, prev_obs.size(1))):
            integrated_intensity[:, i, :] = integrated_intensity[:, i, :] - integrated_intensity[:, i-1, :]
        integrated_intensity = torch.exp(integrated_intensity)

        return F.sigmoid(propensity_pred).cpu(), (F.sigmoid(intensity_pred) / integrated_intensity).cpu()


class SCIPPropensityNetworkHistory(SCIP):

    model_type = 'propensity_history'
    tuning_criterion = 'bce'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = (self.dim_treatments + 1) + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome * 2 if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_treatments

        self._init_specific(args.model.propensity_history)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            assert self.hparams.dataset.treatment_mode == 'multilabel'  # Only binary multilabel regime possible
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        prev_treatments = batch['prev_treatments']
        prev_treatment_times = batch['prev_treatment_times']
        vitals_or_prev_outputs = []
        vitals_or_prev_outputs.append(batch['vitals']) if self.has_vitals else None
        vitals_or_prev_outputs.append(batch['prev_outputs']) if self.autoregressive else None
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        observation_mask = (~torch.isnan(batch['prev_outputs'])).cumsum(dim=1) / batch['prev_outputs'].size(1)
        static_features = batch['static_features']
        x = torch.cat((prev_treatments, prev_treatment_times), dim=-1)
        x = torch.cat((x, vitals_or_prev_outputs), dim=-1)
        x = torch.cat((x, observation_mask), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.cde(x, init_states=None, device=self.device)
        propensity_pred = self.output_layer(x)
        intensity_pred = self.intensity_layer(x)
        return propensity_pred, intensity_pred

    def training_step(self, batch, batch_ind):
        propensity_pred, intensity_pred = self(batch)

        propensity_loss = self.bce_loss(propensity_pred, batch['current_treatments'].double(), kind='predict')
        propensity_loss = ((batch['active_entries'].squeeze(-1) * batch['current_treatment_times'].squeeze(-1) * propensity_loss).sum() /
                           (batch['active_entries']*batch['current_treatment_times']).sum())

        intensity_loss = self.bce_loss(intensity_pred, batch['current_treatment_times'].double(), kind='predict')
        intensity_loss = (batch['active_entries'].squeeze(-1) * intensity_loss).sum() / batch['active_entries'].sum()

        self.log(f'{self.model_type}_propensity_loss', propensity_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_intensity_loss', propensity_loss, on_epoch=True, on_step=False, sync_dist=True)
        return propensity_loss + intensity_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        propensity_pred, intensity_pred = self(batch)

        # Compute the integrated intensity
        prev_treatments = batch['prev_treatments']
        prev_treatment_times = batch['prev_treatment_times']
        vitals_or_prev_outputs = []
        vitals_or_prev_outputs.append(batch['vitals']) if self.has_vitals else None
        vitals_or_prev_outputs.append(batch['prev_outputs']) if self.autoregressive else None
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        observation_mask = (~torch.isnan(batch['prev_outputs'])).cumsum(dim=1) / batch['prev_outputs'].size(1)
        static_features = batch['static_features']
        prev_obs = torch.cat((prev_treatments, prev_treatment_times), dim=-1)
        prev_obs = torch.cat((prev_obs, vitals_or_prev_outputs), dim=-1)
        prev_obs = torch.cat((prev_obs, observation_mask), dim=-1)
        prev_obs = torch.cat((prev_obs, static_features.unsqueeze(1).expand(-1, prev_obs.size(1), -1)), dim=-1)

        def intensity_integrand(t, x):
            # Take last Z_t from CDE
            Z_t = self.cde(prev_obs, init_states=None, t_max=t, device=self.device)[:, -1, :].to(self.device).double()
            return F.sigmoid(self.intensity_layer(Z_t))

        intensity_init = F.sigmoid(self.intensity_layer(self.cde.input_layer(prev_obs[:, 0, :])))
        integrated_intensity = odeint(func=intensity_integrand, y0=intensity_init, method='euler',
                                      t=torch.linspace(0, 1, prev_obs.size(1), device=self.device)).permute(1, 0, 2)

        for i in reversed(range(1, prev_obs.size(1))):
            integrated_intensity[:, i, :] = integrated_intensity[:, i, :] - integrated_intensity[:, i - 1, :]
        integrated_intensity = torch.exp(integrated_intensity)
        return F.sigmoid(propensity_pred).cpu(), (F.sigmoid(intensity_pred) / integrated_intensity).cpu()




class SCIPEncoder(SCIP):

    model_type = 'encoder'
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 propensity_treatment: SCIPPropensityNetworkTreatment = None,
                 propensity_history: SCIPPropensityNetworkHistory = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome * 2 if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_outcome
        self.propensity_treatment = propensity_treatment
        self.propensity_history = propensity_history

        self._init_specific(args.model.encoder)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.dataset_collection is not None and 'sw_tilde_enc' not in self.dataset_collection.train_f.data:
            self.dataset_collection.process_propensity_intensity_train_f(self.propensity_treatment, self.propensity_history,
                                                                         stabilize=self.hparams.exp.stabilize)
            self.dataset_collection.train_f.data['sw_tilde_enc'] = clip_normalize_stabilized_weights(
                self.dataset_collection.train_f.data['stabilized_weights'],
                self.dataset_collection.train_f.data['active_entries'],
                multiple_horizons=False
            )

        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        vitals_or_prev_outputs = []
        vitals_or_prev_outputs.append(batch['vitals']) if self.has_vitals else None
        vitals_or_prev_outputs.append(batch['prev_outputs']) if self.autoregressive else None
        vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        observation_mask = (~torch.isnan(batch['prev_outputs'])).cumsum(dim=1) / batch['prev_outputs'].size(1)
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        x = torch.cat((vitals_or_prev_outputs, curr_treatments), dim=-1)
        x = torch.cat((x, observation_mask), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        r = self.cde(x, init_states=None, device=self.device)
        outcome_pred = self.output_layer(r)
        return outcome_pred, r

    def training_step(self, batch, batch_ind):
        outcome_pred, _ = self(batch)
        outcome_pred[torch.isnan(batch['outputs'])], batch['outputs'][torch.isnan(batch['outputs'])] = 0, 0
        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        mse_loss[torch.isnan(batch['outputs'])] = 0
        mse_loss = (batch['active_entries'] * mse_loss).nansum() / batch['active_entries'].sum()
        self.log(f'{self.model_type}_mse_loss', mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return mse_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        outcome_pred, r = self(batch)
        return outcome_pred.cpu(), r.cpu()

    def get_representations(self, dataset: Dataset) -> np.array:
        logger.info(f'Representations inference for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        _, r = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return r.numpy()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        # Creating Dataloader
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred, _ = [torch.cat(arrs) for arrs in zip(*self.trainer.predict(self, data_loader))]
        return outcome_pred.numpy()


class SCIPDecoder(SCIP):

    model_type = 'decoder'
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 encoder: SCIPEncoder = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 encoder_r_size: int = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome * 2
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_outcome

        self.encoder = encoder
        encoder_r_size = self.encoder.seq_hidden_units if encoder is not None else encoder_r_size

        self._init_specific(args.model.decoder, encoder_r_size=encoder_r_size)
        self.save_hyperparameters(args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder)
        if self.dataset_collection is not None and 'sw_tilde_dec' not in self.dataset_collection.train_f.data:
            self.dataset_collection.train_f.data['stabilized_weights'] = \
                np.cumprod(self.dataset_collection.train_f.data['stabilized_weights'], axis=-1)[:, 1:]
            self.dataset_collection.train_f.data['sw_tilde_dec'] = clip_normalize_stabilized_weights(
                self.dataset_collection.train_f.data['stabilized_weights'],
                self.dataset_collection.train_f.data['active_entries'],
                multiple_horizons=True
            )
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        curr_treatments = batch['current_treatments']
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        init_states = batch['init_state']
        observation_mask = (~torch.isnan(batch['prev_outputs'])).cumsum(dim=1) / batch['prev_outputs'].size(1)
        x = torch.cat((curr_treatments, prev_outputs), dim=-1)
        x = torch.cat((x, observation_mask), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        x = self.cde(x, init_states=self.memory_adapter(init_states), device=self.device)
        outcome_pred = self.output_layer(x)
        return outcome_pred

    def training_step(self, batch, batch_ind):
        outcome_pred = self(batch)
        outcome_pred[torch.isnan(batch['outputs'])], batch['outputs'][torch.isnan(batch['outputs'])] = 0, 0
        mse_loss = F.mse_loss(outcome_pred, batch['outputs'], reduce=False)
        if batch['outputs'].size(1) == 1:
            mse_loss[torch.isnan(batch['prev_outputs'])] = 0
            batch['active_entries'][torch.isnan(batch['prev_outputs'])] = 0
        weighted_mse_loss = mse_loss * batch['sw_tilde_dec'].unsqueeze(-1)
        weighted_mse_loss = (batch['active_entries'] * weighted_mse_loss).sum() / batch['active_entries'].sum()
        self.log(f'{self.model_type}_mse_loss', weighted_mse_loss, on_epoch=True, on_step=False, sync_dist=True)
        return weighted_mse_loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        return self(batch).cpu()

    def get_predictions(self, dataset: Dataset) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)
        outcome_pred = torch.cat(self.trainer.predict(self, data_loader))
        return outcome_pred.numpy()
