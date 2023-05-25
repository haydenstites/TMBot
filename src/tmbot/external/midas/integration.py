import numpy as np
import torch
import os
import cv2

from .midas31.midas.model_loader import load_model

class TMMidas():
    def __init__(self, model_path, optimize = False, height = None, square = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = os.path.splitext(os.path.split(model_path)[-1])[0]
        self.optimize = optimize

        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_path, self.model_type, self.optimize, height, square)

    def step(self, frame : np.ndarray):
        """Performs a step taking a single frame

        Args:
            frame (np.ndarray) : Array representing an image observation

        Returns:
            prediction (np.ndarray) : Array representing an image prediction
        """
        with torch.no_grad():
            original_image_rgb = np.flip(frame.transpose(), 2)
            image = self.transform({"image": original_image_rgb/255})["image"]

            prediction = process(self.device, self.model, self.model_type, image, (self.net_w, self.net_h),
                original_image_rgb.shape[1::-1], self.optimize, False)
            
            return self._depth(prediction, True)
        
    def _depth(self, depth, grayscale, bits=1):
        if not grayscale:
            bits = 1

        if not np.isfinite(depth).all():
            depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")

        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8*bits))-1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

        if not grayscale:
            out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

        return out
    
first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction