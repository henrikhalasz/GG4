from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


PathLike = Union[str, Path]


class BMI_and_Hand:
    def __init__(self, brain):
        """
            brain: an instance of Brain, which has a next_state(input) method and a measure() method.
        """
        self._brain = brain
        self._arm = _Arm()
        BMI_and_Hand.repeat_measurement = 100
        self.ann = load_ann("neural_activity_to_muscle_weights.npz")

    def next_state(self, input=None) -> np.ndarray:
        """
            Apply the input to the brain and return the new neural activity after movement.
        """
        self._brain.next_state(input)
        measurements = []
        for _ in range(BMI_and_Hand.repeat_measurement):
            measurements.append(np.array(self._brain.measure()))
        measurement = np.mean(np.array(measurements), axis=0)  # average over measurements to reduce noise (interal use only, not for students to use)
        self._cal_hand_pos(measurement)
        return np.array(self._brain.measure())
    
    @property
    def hand_pos(self):
        """Return the current hand position as a numpy array of shape (2,)."""
        return self._arm.hand_pos

    def _cal_hand_pos(self, measurement):
        muscle = self.ann.step(torch.tensor(measurement, dtype=torch.float32))
        self._arm.move(*muscle)


def _resolve_weights_path(weights_path: PathLike) -> Path:
    path = Path(weights_path)
    if path.exists():
        return path
    if not path.is_absolute():
        beside_this_file = Path(__file__).resolve().with_name(path.name)
        if beside_this_file.exists():
            return beside_this_file
    return path


class _FirstStageDecoder(nn.Module):
    """
    Provided starting head, kept as the first stage.

    Input shape:
        (..., 16)

    Output shape:
        (..., 2)

    Convention used by the muscle head:
        decoded[..., 0] = x1 = frequency / muscle-selection rhythm
        decoded[..., 1] = x2 = power rhythm
    """

    def __init__(
        self,
        weights_path: PathLike = "neural_activity_to_muscle_weights.npz",
        *,
        dtype: torch.dtype = torch.float32,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        weights = np.load(_resolve_weights_path(weights_path))

        encoder_weight = torch.as_tensor(weights["encoder_weight"], dtype=dtype)
        output_weight = torch.as_tensor(weights["output_weight"], dtype=dtype)
        output_bias = torch.as_tensor(weights["output_bias"], dtype=dtype)

        self.feature_encoder = nn.Linear(16, 8, bias=False, dtype=dtype)
        self.activation = nn.Identity()
        self.output_layer = nn.Linear(8, 2, bias=True, dtype=dtype)

        with torch.no_grad():
            self.feature_encoder.weight.copy_(encoder_weight)
            self.output_layer.weight.copy_(output_weight)
            self.output_layer.bias.copy_(output_bias)

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, neural_activity: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder(neural_activity)
        features = self.activation(features)
        return self.output_layer(features)


class _LatentToMuscleHead(nn.Module):
    """
    Stateful 2 -> 4 muscle activation head.

    Input convention:
        latent[..., 0] = x1 = frequency / muscle-selection rhythm
        latent[..., 1] = x2 = power rhythm

    Output:
        muscle activation rates in [0, 1], shape (..., 4)

    The band-pass filters are fixed torch.nn.Conv1d layers. Their numerical
    kernels are loaded from the NPZ file. No frequency constants are needed in
    this Python module.
    """

    def __init__(
        self,
        weights_path: PathLike = "neural_activity_to_muscle_weights.npz",
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        weights = np.load(_resolve_weights_path(weights_path))

        cos_kernel = torch.as_tensor(weights["muscle_filter_cos"], dtype=dtype)
        sin_kernel = torch.as_tensor(weights["muscle_filter_sin"], dtype=dtype)
        if cos_kernel.ndim != 2 or sin_kernel.shape != cos_kernel.shape:
            raise ValueError("muscle_filter_cos and muscle_filter_sin must both have shape (4, kernel_size).")
        if cos_kernel.shape[0] != 4:
            raise ValueError("This head expects exactly four muscle filters.")

        self.output_dim = int(cos_kernel.shape[0])
        self.kernel_size = int(cos_kernel.shape[1])

        self.cos_bank = nn.Conv1d(1, self.output_dim, self.kernel_size, bias=False, dtype=dtype)
        self.sin_bank = nn.Conv1d(1, self.output_dim, self.kernel_size, bias=False, dtype=dtype)
        with torch.no_grad():
            self.cos_bank.weight.copy_(cos_kernel[:, None, :])
            self.sin_bank.weight.copy_(sin_kernel[:, None, :])
        for parameter in self.cos_bank.parameters():
            parameter.requires_grad_(False)
        for parameter in self.sin_bank.parameters():
            parameter.requires_grad_(False)

        def scalar(name: str) -> torch.Tensor:
            return torch.as_tensor(weights[name], dtype=dtype).reshape(())

        self.register_buffer("selector_threshold", torch.as_tensor(weights["selector_threshold"], dtype=dtype))
        self.register_buffer("selector_scale", torch.as_tensor(weights["selector_scale"], dtype=dtype))
        self.register_buffer("selector_gain", scalar("selector_gain"))

        self.register_buffer("power_positive_threshold", scalar("power_positive_threshold"))
        self.register_buffer("power_negative_threshold", scalar("power_negative_threshold"))
        self.register_buffer("power_wake_sharpness", scalar("power_wake_sharpness"))
        self.register_buffer("power_peak_threshold", scalar("power_peak_threshold"))

        self.power_half_period_steps = int(np.asarray(weights["power_half_period_steps"]).item())
        self.power_full_period_steps = int(np.asarray(weights["power_full_period_steps"]).item())
        if not (1 <= self.power_half_period_steps < self.kernel_size):
            raise ValueError("power_half_period_steps must be in [1, kernel_size).")
        if not (1 <= self.power_full_period_steps < self.kernel_size):
            raise ValueError("power_full_period_steps must be in [1, kernel_size).")

        self.register_buffer("power_half_pair_weight", scalar("power_half_pair_weight"))
        self.register_buffer("power_full_pair_weight", scalar("power_full_pair_weight"))
        self.register_buffer("power_evidence_threshold", scalar("power_evidence_threshold"))
        self.register_buffer("power_evidence_hit_decay", scalar("power_evidence_hit_decay"))
        self.register_buffer("power_evidence_miss_decay", scalar("power_evidence_miss_decay"))
        self.register_buffer("power_evidence_add_rate", scalar("power_evidence_add_rate"))
        self.register_buffer("power_drive_threshold", scalar("power_drive_threshold"))
        self.register_buffer("power_drive_scale", scalar("power_drive_scale"))
        self.register_buffer("power_enter_rate", scalar("power_enter_rate"))
        self.register_buffer("power_exit_rate", scalar("power_exit_rate"))

        self.register_buffer("latent_cache", torch.zeros(1, self.kernel_size, 2, dtype=dtype))
        self.register_buffer("power_state", torch.zeros(1, dtype=dtype))
        self.register_buffer("evidence_state", torch.zeros(1, dtype=dtype))

    def reset_state(self, batch_size: int = 1) -> None:
        device = self.selector_threshold.device
        dtype = self.selector_threshold.dtype
        self.latent_cache = torch.zeros(batch_size, self.kernel_size, 2, device=device, dtype=dtype)
        self.power_state = torch.zeros(batch_size, device=device, dtype=dtype)
        self.evidence_state = torch.zeros(batch_size, device=device, dtype=dtype)

    def step(
        self,
        latent_t: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        unbatched = latent_t.ndim == 1
        if unbatched:
            latent_t = latent_t.unsqueeze(0)
        if latent_t.ndim != 2 or latent_t.shape[-1] != 2:
            raise ValueError(f"Expected latent_t shape (2,) or (batch, 2), got {tuple(latent_t.shape)}")

        latent_t = latent_t.to(device=self.selector_threshold.device, dtype=self.selector_threshold.dtype)
        batch_size = latent_t.shape[0]
        if self.latent_cache.shape[0] != batch_size:
            self.reset_state(batch_size=batch_size)

        new_cache = torch.cat([self.latent_cache[:, 1:, :], latent_t[:, None, :]], dim=1)
        activation, new_power, new_evidence, debug = self._compute_from_cache(
            new_cache,
            self.power_state,
            self.evidence_state,
        )

        self.latent_cache = new_cache.detach()
        self.power_state = new_power.detach()
        self.evidence_state = new_evidence.detach()

        if unbatched:
            activation = activation.squeeze(0)
            debug = {
                key: value.squeeze(0) if value.ndim > 0 and value.shape[0] == 1 else value
                for key, value in debug.items()
            }
        if return_debug:
            return activation, debug
        return activation

    def forward(
        self,
        latents: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        unbatched = latents.ndim == 2
        if unbatched:
            latents = latents.unsqueeze(0)
        if latents.ndim != 3 or latents.shape[-1] != 2:
            raise ValueError(f"Expected latents shape (time, 2) or (batch, time, 2), got {tuple(latents.shape)}")

        latents = latents.to(device=self.selector_threshold.device, dtype=self.selector_threshold.dtype)
        batch_size, time_steps, _ = latents.shape
        cache = torch.zeros(batch_size, self.kernel_size, 2, device=latents.device, dtype=latents.dtype)
        power = torch.zeros(batch_size, device=latents.device, dtype=latents.dtype)
        evidence = torch.zeros(batch_size, device=latents.device, dtype=latents.dtype)

        outputs = []
        debug_lists: Dict[str, list[torch.Tensor]] = {
            "latent": [],
            "band_energy": [],
            "selector": [],
            "positive_wake": [],
            "negative_wake": [],
            "half_period_pair": [],
            "full_period_pair": [],
            "period_evidence": [],
            "evidence_memory": [],
            "power": [],
        }

        for t in range(time_steps):
            cache = torch.cat([cache[:, 1:, :], latents[:, t:t + 1, :]], dim=1)
            activation, power, evidence, debug = self._compute_from_cache(cache, power, evidence)
            outputs.append(activation)
            if return_debug:
                for key in debug_lists:
                    debug_lists[key].append(debug[key])

        output = torch.stack(outputs, dim=1)
        if not return_debug:
            return output.squeeze(0) if unbatched else output

        debug_stacked = {key: torch.stack(values, dim=1) for key, values in debug_lists.items()}
        if unbatched:
            output = output.squeeze(0)
            debug_stacked = {key: value.squeeze(0) for key, value in debug_stacked.items()}
        return output, debug_stacked

    def _wake_pair(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positive = torch.sigmoid(self.power_wake_sharpness * (value - self.power_positive_threshold))
        negative = torch.sigmoid(self.power_wake_sharpness * (-value - self.power_negative_threshold))
        return positive, negative

    def _compute_from_cache(
        self,
        cache: torch.Tensor,
        power: torch.Tensor,
        evidence_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # x1 controls frequency/muscle selection.
        selection_history = cache[:, :, 0].unsqueeze(1)
        cos_response = self.cos_bank(selection_history).squeeze(-1)
        sin_response = self.sin_bank(selection_history).squeeze(-1)
        band_energy = torch.sqrt(cos_response.square() + sin_response.square() + 1e-12)

        selector_raw = (band_energy - self.selector_threshold) / (self.selector_scale + 1e-12)
        selector = torch.clamp(self.selector_gain * selector_raw, 0.0, 1.0)

        # x2 controls power.
        power_history = cache[:, :, 1]
        current_value = power_history[:, -1]
        half_value = power_history[:, -1 - self.power_half_period_steps]
        full_value = power_history[:, -1 - self.power_full_period_steps]

        positive_now, negative_now = self._wake_pair(current_value)
        positive_half, negative_half = self._wake_pair(half_value)
        positive_full, negative_full = self._wake_pair(full_value)

        half_period_pair = positive_now * negative_half + negative_now * positive_half
        full_period_pair = positive_now * positive_full + negative_now * negative_full

        period_evidence = (
            self.power_half_pair_weight * half_period_pair
            + self.power_full_pair_weight * full_period_pair
        )
        period_evidence = torch.clamp(period_evidence, 0.0, 1.0)

        current_peak = torch.maximum(positive_now, negative_now)
        evidence_hit = (period_evidence > self.power_evidence_threshold) & (current_peak > self.power_peak_threshold)
        strengthened = self.power_evidence_hit_decay * evidence_memory + self.power_evidence_add_rate * period_evidence
        faded = self.power_evidence_miss_decay * evidence_memory
        new_evidence = torch.where(evidence_hit, strengthened, faded)
        new_evidence = torch.clamp(new_evidence, 0.0, 1.0)

        drive = torch.clamp(
            (new_evidence - self.power_drive_threshold) / (self.power_drive_scale + 1e-12),
            0.0,
            1.0,
        )
        rate = torch.where(drive > power, self.power_enter_rate, self.power_exit_rate)
        new_power = torch.clamp((1.0 - rate) * power + rate * drive, 0.0, 1.0)

        activation = torch.clamp(new_power[:, None] * selector, 0.0, 1.0)

        debug = {
            "latent": cache[:, -1, :],
            "band_energy": band_energy,
            "selector": selector,
            "positive_wake": positive_now,
            "negative_wake": negative_now,
            "half_period_pair": half_period_pair,
            "full_period_pair": full_period_pair,
            "period_evidence": period_evidence,
            "evidence_memory": new_evidence,
            "power": new_power,
        }
        return activation, new_power, new_evidence, debug


class _NeuralActivityToMuscleANN(nn.Module):
    """
    Complete 16 -> 4 ANN.

    16D neural activity
        -> FirstStageDecoder
        -> x1/x2 latents
        -> four muscle activation rates in [0, 1]
    """

    def __init__(
        self,
        weights_path: PathLike = "neural_activity_to_muscle_weights.npz",
        *,
        dtype: torch.dtype = torch.float32,
        freeze_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.first_stage = _FirstStageDecoder(weights_path, dtype=dtype, freeze=freeze_decoder)
        self.muscle_head = _LatentToMuscleHead(weights_path, dtype=dtype)

    def reset_state(self, batch_size: int = 1) -> None:
        self.muscle_head.reset_state(batch_size=batch_size)

    def decode_latent(self, neural_activity: torch.Tensor) -> torch.Tensor:
        return self.first_stage(neural_activity)

    def step(
        self,
        neural_activity_t: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        latent_t = self.first_stage(neural_activity_t)
        return self.muscle_head.step(latent_t, return_debug=return_debug)

    def forward(
        self,
        neural_activity: torch.Tensor,
        *,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        latent = self.first_stage(neural_activity)
        return self.muscle_head(latent, return_debug=return_debug)


def load_ann(
    weights_path: PathLike = "neural_activity_to_muscle_weights.npz",
    *,
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.float32,
    freeze_decoder: bool = True,
) -> _NeuralActivityToMuscleANN:
    ann = _NeuralActivityToMuscleANN(weights_path, dtype=dtype, freeze_decoder=freeze_decoder)
    if device is not None:
        ann = ann.to(device)
    return ann


class _Arm:
    def __init__(self):
        _Arm.forearm_length = 30
        _Arm.upperarm_length = 30
        _Arm._elbow_friction_dry = 0.05
        _Arm._elbow_friction_wet = 5
        _Arm._shoulder_friction_dry = 0.05
        _Arm._shoulder_friction_wet = 7.5
        self._shoulder_angle = 0  # radians
        self._elbow_angle = 0  # radians


    @property
    def hand_pos(self):
        x = _Arm.upperarm_length * np.cos(self._shoulder_angle) + _Arm.forearm_length * np.cos(self._shoulder_angle + self._elbow_angle)
        y = _Arm.upperarm_length * np.sin(self._shoulder_angle) + _Arm.forearm_length * np.sin(self._shoulder_angle + self._elbow_angle)
        return np.array([x, y])
    
    @property
    def _elbow_pos(self):
        x = _Arm.upperarm_length * np.cos(self._shoulder_angle)
        y = _Arm.upperarm_length * np.sin(self._shoulder_angle)
        return np.array([x, y])

    def move(
            self,
            shoulder_pos_muscle: float,
            shoulder_neg_muscle: float,
            elbow_pos_muscle: float,
            elbow_neg_muscle: float,
    ):
        """The muscle activation is between 0 and 1, where 0 means no activation and 1 means full activation."""
        shoulder_pos_power = shoulder_pos_muscle * self._max_power_at_angle(self._shoulder_angle, "shoulder_pos")
        shoulder_neg_power = shoulder_neg_muscle * self._max_power_at_angle(self._shoulder_angle, "shoulder_neg")
        elbow_pos_power = elbow_pos_muscle * self._max_power_at_angle(self._elbow_angle, "elbow_pos")
        elbow_neg_power = elbow_neg_muscle * self._max_power_at_angle(self._elbow_angle, "elbow_neg")
        
        if np.abs(shoulder_pos_power - shoulder_neg_power) > _Arm._shoulder_friction_dry:
            shoulder_net_speed = (shoulder_pos_power - shoulder_neg_power - np.sign(shoulder_pos_power - shoulder_neg_power) * _Arm._shoulder_friction_dry) / _Arm._shoulder_friction_wet
        else:
            shoulder_net_speed = 0

        if np.abs(elbow_pos_power - elbow_neg_power) > _Arm._elbow_friction_dry:
            elbow_net_speed = (elbow_pos_power - elbow_neg_power - np.sign(elbow_pos_power - elbow_neg_power) * _Arm._elbow_friction_dry) / _Arm._elbow_friction_wet
        else:
            elbow_net_speed = 0

        self._shoulder_angle += shoulder_net_speed
        self._elbow_angle += elbow_net_speed

        
    @staticmethod
    def _max_power_at_angle(angle: float, muscle: str):
        """The maximum power that the muscle can generate at a given joint angle. It is a function of the joint angle."""
        if muscle == "shoulder_pos":
            return 1
        elif muscle == "shoulder_neg":
            return 1
        elif muscle == "elbow_pos":
            return 1
        elif muscle == "elbow_neg":
            return 1
    
