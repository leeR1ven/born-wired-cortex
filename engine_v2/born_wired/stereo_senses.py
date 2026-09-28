"""Two raw RGB retinas, plus a separate offline stereo diagnostic.

Only RawEyes belongs in the controller sensory path. The separate StereoSenses
class is an offline diagnostic with programmed matching and color extraction;
its computed features must not be fed to the neural controller. RawEyes performs
no calibration or feature calculation and does not depend on StereoSenses.
"""

import mujoco
import numpy as np
from scipy import sparse

from . import torch_execution
from .synapses import _scalar


class RawEyes:
    """Two rendered RGB camera images, without calibration or interpretation.

    The sheet is a partition of the drawn picture: the cells are laid out on
    the picture itself, closer together near the middle of the view and further
    apart towards the rim, and each cell reports the mean of the drawn pixels
    under its own slice. Nothing is stretched, squeezed or interpolated: the
    middle is simply answered for by more cells than the rim, so the middle
    resolves more of the drawn picture and the rim resolves less. Sampling
    density falling away from the middle is the whole of the eye's own wiring.
    Averaging light over an acceptance angle is what a receptor does; nothing
    here estimates a distance, an object or a match.
    """

    def __init__(self, body, width=48, height=36, oversample=None, centre_gain=2.5):
        if (isinstance(width, (bool, np.bool_)) or isinstance(height, (bool, np.bool_))
                or not isinstance(width, (int, np.integer))
                or not isinstance(height, (int, np.integer)) or width < 1 or height < 1):
            raise ValueError("width/height must be positive integers")
        if not np.isfinite(centre_gain) or float(centre_gain) < 1.:
            raise ValueError("centre_gain must be a finite number of at least one")
        if oversample is None:
            # The drawn picture only has to be fine enough for the finest part
            # of the sheet, and that is the middle, where the cells stand
            # `centre_gain` times closer together than they would on equal
            # slices. Drawing it finer than that costs render time and shows
            # nowhere.
            oversample = max(2, int(np.ceil(centre_gain)))
        if (isinstance(oversample, (bool, np.bool_)) or not isinstance(oversample, (int, np.integer))
                or oversample < 1):
            raise ValueError("oversample must be a positive whole number")
        if not isinstance(getattr(body, "model", None), mujoco.MjModel) or not isinstance(getattr(body, "data", None), mujoco.MjData):
            raise ValueError("body must expose a MuJoCo model and data")
        self.body = body
        self.width, self.height = int(width), int(height)
        self.oversample = int(oversample)
        self.centre_gain = float(centre_gain)
        self._camera = tuple(mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                             for name in ("eye_left", "eye_right"))
        if min(self._camera) < 0:
            raise ValueError("model requires eye_left and eye_right cameras")
        drawn = (self.height*self.oversample, self.width*self.oversample)
        self._renderer = mujoco.Renderer(body.model, height=drawn[0], width=drawn[1])
        self._rows = self._patches(self.height, drawn[0])
        self._columns = self._patches(self.width, drawn[1])
        self._row_weights, self._column_weights, self._patch_area = self._sampling(drawn)
        self._device, self._device_operators = self._device_operators_for()
        self._closed = False

    def _device_operators_for(self):
        """The same two operators on the graphics device, when there is one.

        Reading a cell's patch is two sums against runs of ones. Every addend
        is a whole number no larger than 255 and every partial sum is a whole
        number below 2**24, so each sum is an exact float32 no matter the order
        its terms are added in - which is what lets a dense block product on a
        device answer with the host's own integers rather than with something
        near them. The two operators are mostly zeros and only a few hundred
        entries wide, so the dense form costs nothing to hold.

        tf32 is switched off for this process when it is on: it keeps ten
        mantissa bits, which would make the products inexact and the claim
        above false. Nothing in this project wanted it on.
        """
        device = torch_execution.resolve()
        if device is None:
            return None, None
        try:
            torch = torch_execution.torch_module()
            if device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = False
            operators = (
                torch.as_tensor(self._row_weights.toarray(), dtype=torch.float32,
                                device=device),
                torch.as_tensor(self._column_weights.toarray(), dtype=torch.float32,
                                device=device),
                torch.as_tensor(self._patch_area, dtype=torch.float64, device=device),
            )
        except Exception:
            # A device that cannot hold two small tables is a device this sensor
            # does not use; the host answer is the same answer.
            return None, None
        return device, operators

    def _patches(self, count, drawn):
        """The drawn rows or columns each retinal row or column answers for.

        The `count` cells start on equal slices of the view, and those slices
        are then pulled toward the middle, so that `centre_gain` times as many
        cells sit on a degree at the middle of the view as at the rim. Each
        cell answers for the drawn pixels of its own slice - narrow at the
        middle, wide at the rim - and the slices tile the drawn picture
        exactly, so the sheet is that picture read at a density falling off
        from the middle, and nothing on it is displaced.
        """
        t = ((np.arange(count) + .5)/count)*2. - 1.
        k = 1./self.centre_gain
        p = t*(k + (1. - k)*t*t)
        centres = (p + 1.)*.5*drawn
        widths = np.maximum(np.abs(np.gradient(centres)), 1.)
        low = np.clip(np.rint(centres - .5*widths).astype(int), 0, drawn)
        high = np.clip(np.rint(centres + .5*widths).astype(int), 0, drawn)
        return low, np.maximum(high, low + 1)

    def _report(self, picture):
        """Mean of the drawn picture over each cell's own patch."""
        if self._device_operators is not None:
            return self._report_on_device(picture)
        # A patch is a rectangle, so its sum is a sum down its rows and then
        # across its columns, and each of those is one product against the
        # 0/1 operators built in _sampling. Every entry and every partial sum
        # is a whole number well inside single precision (see _sampling), so
        # this returns exactly what running totals over both axes returned.
        # The operators are sparse because each cell covers a run of only a
        # few rows or columns; multiplying against them touches the runs
        # rather than the whole drawn picture, and unlike a dense product it
        # starts no thread pool, which matters here because the drawing call
        # that follows has to get back on the same cores.
        height, width = picture.shape[:2]
        rows = np.asarray(self._row_weights
                          @ np.asarray(picture).reshape(height, width*3).astype(np.float32))
        columns = rows.reshape(self.height, width, 3).transpose(0, 2, 1)
        columns = columns.reshape(self.height*3, width)
        totals = np.asarray((self._column_weights @ columns.T).T)
        totals = totals.reshape(self.height, 3, self.width).transpose(0, 2, 1)
        return np.rint(totals/self._patch_area).astype(np.uint8)

    def _report_on_device(self, picture):
        """The same two patch sums, evaluated on the graphics device.

        The order of the terms does not enter the answer - see
        _device_operators_for - so a dense block product over whole rows is the
        same integer result the host reaches by touching only the runs. What it
        saves is the host walking three quarters of a million bytes twice per
        eye per step to convert, sum and round them, which is several times the
        drawing call that produced them.
        """
        row_weights, column_weights, patch_area = self._device_operators
        torch = torch_execution.torch_module()
        height, width = picture.shape[:2]
        # The bytes go over as the one-byte channels they are drawn as and are
        # widened where there is room: handing the copy a float32 destination
        # makes it convert three quarters of a million times as it crawls
        # through pageable memory, and that costs more than the sums do.
        drawn = torch.as_tensor(picture, device=self._device).to(torch.float32)
        rows = row_weights @ drawn.reshape(height, width*3)
        columns = rows.reshape(self.height, width, 3).permute(0, 2, 1)
        columns = columns.reshape(self.height*3, width)
        totals = (column_weights @ columns.T).T.to(torch.float64)
        totals = totals.reshape(self.height, 3, self.width).permute(0, 2, 1)
        return torch.round(totals/patch_area).to(torch.uint8).cpu().numpy()

    def _sampling(self, drawn):
        """The two 0/1 operators the patch sums are read with, and patch areas.

        The rows and columns of the sheet are independent: a patch is the
        product of one row interval and one column interval, so the two
        intervals can be applied one after the other as matrix products. The
        operator covering a cell is a run of ones from its low edge up to its
        high edge, which is what _patches already answers for.

        Single precision is exact here rather than merely close. Every drawn
        pixel is a whole number no larger than 255, so a sum of whole numbers
        is a whole number; a float32 counts by ones up to 2**24, and the
        largest patch holds fewer pixels than that, so no partial sum ever
        rounds. The guard below refuses a sheet whose picture is drawn so
        coarsely against so fine a middle that a patch could reach 2**24/255
        pixels, which would make the claim false.
        """
        drawn_rows, drawn_columns = drawn
        low_r, high_r = self._rows
        low_c, high_c = self._columns
        largest = int(np.max(high_r - low_r))*int(np.max(high_c - low_c))
        if largest*255 >= 2**24:
            raise ValueError('a patch is too large to sum exactly in single precision')
        counts_r, counts_c = high_r - low_r, high_c - low_c
        row_weights = sparse.csr_matrix(
            (np.ones(int(counts_r.sum()), np.float32),
             (np.repeat(np.arange(self.height, dtype=np.int32), counts_r),
              np.concatenate([np.arange(low_r[i], high_r[i], dtype=np.int32)
                              for i in range(self.height)]))),
            shape=(self.height, drawn_rows))
        column_weights = sparse.csr_matrix(
            (np.ones(int(counts_c.sum()), np.float32),
             (np.repeat(np.arange(self.width, dtype=np.int32), counts_c),
              np.concatenate([np.arange(low_c[j], high_c[j], dtype=np.int32)
                              for j in range(self.width)]))),
            shape=(self.width, drawn_columns))
        area = (counts_r[:, None]*counts_c[None, :])[:, :, None].astype(np.float64)
        return row_weights, column_weights, area

    def observe_raw(self):
        """Return uint8 RGB (left/right, height, width, channels) only."""
        if self._closed:
            raise RuntimeError("raw eyes are closed")
        frames = []
        for camera in self._camera:
            self._renderer.update_scene(self.body.data, camera=camera)
            frames.append(self._report(self._renderer.render()))
        return np.stack(frames)

    def close(self):
        if not self._closed:
            self._renderer.close()
            self._device, self._device_operators = None, None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class StereoSenses:
    """Eyes L/R; sectors image L/C/R; opponent channels R/G/B.

    Range is optical-axis distance, not Euclidean distance. Missing range is
    represented by ``valid=False, confidence=0, distance=max_range``. Consumers
    must gate range-derived inputs with valid/confidence. Uniform or repetitive
    textures and occlusion deliberately produce missing observations.
    """

    def __init__(self, body, width=96, height=64, max_range=3.):
        if (isinstance(width, bool) or isinstance(height, bool)
                or not isinstance(width, (int, np.integer))
                or not isinstance(height, (int, np.integer))
                or width < 48 or height < 32):
            raise ValueError("width/height must be integers of at least 48/32")
        max_range = _scalar(max_range, "max_range", positive=True)
        if not isinstance(getattr(body, "model", None), mujoco.MjModel) or not isinstance(getattr(body, "data", None), mujoco.MjData):
            raise ValueError("body must expose a MuJoCo model and data")
        self.body = body
        self.width, self.height = int(width), int(height)
        self.max_range = float(max_range)
        self._camera = np.array([mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                                 for name in ("eye_left", "eye_right")])
        if np.any(self._camera < 0):
            raise ValueError("model requires eye_left and eye_right cameras")
        model = body.model
        left, right = self._camera
        if (model.cam_bodyid[left] != model.cam_bodyid[right]
                or np.any(model.cam_mode[self._camera] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
                or np.any(model.cam_projection[self._camera] != mujoco.mjtProjection.mjPROJ_PERSPECTIVE)):
            raise ValueError("eyes must be fixed perspective cameras on the same body")
        fovy = model.cam_fovy[self._camera]
        if not np.allclose(fovy, fovy[0], rtol=0, atol=1e-9) or not 0 < fovy[0] < 170:
            raise ValueError("eyes require the same valid field of view")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.cam_quat[left])
        rotation = rotation.reshape(3, 3)
        baseline = rotation.T @ (model.cam_pos[right] - model.cam_pos[left])
        if (not np.isclose(abs(np.dot(model.cam_quat[left], model.cam_quat[right])), 1., atol=1e-8)
                or baseline[0] <= 0 or not np.allclose(baseline[1:], 0, atol=1e-7)):
            raise ValueError("eyes must be parallel and rectified with a positive horizontal baseline")
        self.baseline = float(baseline[0])
        self.focal_px = self.height / (2 * np.tan(np.deg2rad(fovy[0]) / 2))
        self._rows = np.rint(np.linspace(7, self.height - 8, 4)).astype(int)
        self._columns = np.rint(np.linspace(7, self.width - 8, 8)).astype(int)
        self._max_disparity = min(self.width // 2, 40)
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._closed = False

    @staticmethod
    def _costs(reference, other, v, u, candidates, direction):
        """RGB patch SAD; direction=-1 for left-to-right correspondence."""
        radius = 4
        patch = reference[v-radius:v+radius+1, u-radius:u+radius+1]
        columns = u + direction * candidates
        candidates = candidates[(columns >= radius) & (columns < other.shape[1] - radius)]
        if len(candidates) == 0:
            return candidates, np.empty(0), patch
        patches = np.stack([other[v-radius:v+radius+1,
                                 u+direction*d-radius:u+direction*d+radius+1]
                            for d in candidates])
        costs = np.mean(np.abs(patches - patch), axis=(1, 2, 3))
        return candidates, costs, patch

    def _match(self, left, right):
        disparity = np.zeros((4, 8), dtype=float)
        confidence = np.zeros_like(disparity)
        valid = np.zeros_like(disparity, dtype=bool)
        distance = np.full_like(disparity, self.max_range)
        candidates = np.arange(0, self._max_disparity + 1)
        minimum_disparity = self.focal_px * self.baseline / self.max_range
        for i, v in enumerate(self._rows):
            for j, u in enumerate(self._columns):
                choices, costs, patch = self._costs(left, right, v, u, candidates, -1)
                # Texture must be spatial, not merely RGB channel differences.
                texture = float(np.max(np.std(patch, axis=(0, 1))))
                if len(choices) < 4 or texture < .035:
                    continue
                best = int(np.argmin(costs))
                d = int(choices[best])
                cost = float(costs[best])
                if d == self._max_disparity:
                    continue
                alternatives = costs[np.abs(choices - d) > 1]
                if len(alternatives) == 0:
                    continue
                margin = float(np.min(alternatives) - cost)
                if cost > .14 or margin < .018 or margin < .20 * max(cost, .01):
                    continue
                reverse_choices, reverse_costs, right_patch = self._costs(right, left, v, u-d, candidates, 1)
                if (np.max(np.std(right_patch, axis=(0, 1))) < .035
                        or abs(int(reverse_choices[np.argmin(reverse_costs)]) - d) > 1):
                    continue
                refined = float(d)
                if 0 < best < len(costs) - 1:
                    curvature = costs[best-1] - 2 * cost + costs[best+1]
                    if curvature > 1e-8:
                        refined += float(np.clip(.5 * (costs[best-1] - costs[best+1]) / curvature, -.5, .5))
                if refined < minimum_disparity or refined <= 0:
                    continue
                distance[i, j] = self.focal_px * self.baseline / refined
                disparity[i, j] = refined
                confidence[i, j] = np.clip(margin / .12, 0, 1) * np.clip(1 - cost / .14, 0, 1)
                valid[i, j] = True
        return disparity, confidence, distance, valid

    @staticmethod
    def _retinal_channels(rgb):
        opponent = np.zeros((2, 3, 3), dtype=float)
        brightness = np.zeros((2, 3), dtype=float)
        contrast = np.zeros((2, 3), dtype=float)
        for eye, frame in enumerate(rgb):
            for sector, region in enumerate(np.array_split(frame.astype(float) / 255, 3, axis=1)):
                luminance = region @ np.array([.2126, .7152, .0722])
                # Rectified continuous color opponent currents, without classes.
                currents = np.maximum(1.5 * region - .5 * region.sum(axis=2, keepdims=True), 0).reshape(-1, 3)
                # Retain the strongest 10% chromatic pixels, so a small colored
                # surface is not averaged away by a large achromatic background.
                count = max(1, int(np.ceil(len(currents) * .10)))
                selected = np.argpartition(np.max(currents, axis=1), -count)[-count:]
                opponent[eye, sector] = currents[selected].mean(axis=0)
                brightness[eye, sector] = luminance.mean()
                contrast[eye, sector] = min(1., 2 * luminance.std())
        return opponent, brightness, contrast

    def observe_raw(self):
        """Control input: uint8 RGB array (left/right, height, width, channels).

        This only renders the cameras. It does not match, pool, estimate range,
        compute color channels, or select actions.
        """
        if self._closed:
            raise RuntimeError("stereo sensor is closed")
        frames = []
        for camera in self._camera:
            self._renderer.update_scene(self.body.data, camera=int(camera))
            frames.append(self._renderer.render().copy())
        return np.stack(frames)

    def observe(self):
        """Offline diagnostic only; never feed computed features to the brain."""
        rgb = self.observe_raw()
        normalized = rgb.astype(np.float32) / 255
        disparity, confidence, distance, valid = self._match(*normalized)
        opponent, brightness, contrast = self._retinal_channels(rgb)
        return dict(rgb=rgb, disparity=disparity, confidence=confidence,
                    distance=distance, valid=valid, color_opponent=opponent,
                    brightness=brightness, contrast=contrast)

    def close(self):
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
