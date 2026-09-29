# Copyright (c) OpenMMLab. All rights reserved.
"""Pixel transforms used by the FastVR degradation pipeline."""

import cv2
import numpy as np
import torch


def img2tensor(images, bgr2rgb=True):
    """Convert one image or a list of images from HWC arrays to CHW tensors."""

    def convert(image):
        if image.shape[2] == 3 and bgr2rgb:
            if image.dtype == np.float64:
                image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image.transpose(2, 0, 1)).float()

    if isinstance(images, list):
        return [convert(image) for image in images]
    return convert(images)


class Clip:
    def __init__(self, keys, minimum=0, maximum=255):
        self.keys = keys
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, results):
        for key in self.keys:
            values = results[key]
            if isinstance(values, np.ndarray):
                results[key] = np.clip(values, self.minimum, self.maximum)
            else:
                results[key] = [
                    np.clip(value, self.minimum, self.maximum) for value in values
                ]
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(minimum={self.minimum}, "
            f"maximum={self.maximum})"
        )


class UnsharpMasking:
    def __init__(self, params, keys):
        kernel_size = params["kernel_size"]
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.params = params
        self.keys = keys
        self.weight = params["weight"]
        self.threshold = params["threshold"]
        kernel = cv2.getGaussianKernel(kernel_size, params["sigma"])
        self.kernel = kernel @ kernel.T

    def _apply(self, images):
        single_image = isinstance(images, np.ndarray)
        if single_image:
            images = [images]

        outputs = []
        for image in images:
            image = image.astype(np.float32)
            residue = image - cv2.filter2D(image, -1, self.kernel)
            mask = np.float32(np.abs(residue) > self.threshold)
            soft_mask = cv2.filter2D(mask, -1, self.kernel)
            sharpened = np.clip(image + self.weight * residue, 0, 255)
            outputs.append(soft_mask * sharpened + (1 - soft_mask) * image)
        return outputs[0] if single_image else outputs

    def __call__(self, results):
        if np.random.uniform() <= self.params.get("prob", 1):
            for key in self.keys:
                results[key] = self._apply(results[key])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(params={self.params}, keys={self.keys})"


class RescaleToZeroOne:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            values = results[key]
            if isinstance(values, list):
                results[key] = [value.astype(np.float32) / 255 for value in values]
            else:
                results[key] = values.astype(np.float32) / 255
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self.keys})"
