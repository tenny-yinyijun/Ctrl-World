import os
import numpy as np
import torch
import cv2
from typing import List, Dict, Union, Optional
from pathlib import Path
import warnings

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("LPIPS not available. Install with: pip install lpips")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    warnings.warn("scikit-image not available. Install with: pip install scikit-image")

try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available for FID/FVD. Install with: pip install scipy")


class VideoMetric:
    """
    A class for computing various video quality metrics including:
    - MSE: Mean Squared Error
    - SSIM: Structural Similarity Index
    - LPIPS: Learned Perceptual Image Patch Similarity
    - FID: Fr�chet Inception Distance
    - FVD: Fr�chet Video Distance
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize VideoMetric class.

        Args:
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.device = device
        self.lpips_model = None
        self.i3d_model = None

    def _load_lpips_model(self):
        """Load LPIPS model lazily."""
        if not LPIPS_AVAILABLE:
            raise ImportError("LPIPS not available. Install with: pip install lpips")
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_model.eval()
        return self.lpips_model

    def _load_i3d_model(self):
        """Load I3D model for FVD computation lazily."""
        if self.i3d_model is None:
            # Placeholder for I3D model loading
            # In practice, you would load a pretrained I3D model here
            raise NotImplementedError("I3D model loading not implemented. Use a pretrained I3D model.")
        return self.i3d_model

    def load_video(self, video_path: str) -> np.ndarray:
        """
        Load video from file path.

        Args:
            video_path: Path to the video file

        Returns:
            np.ndarray: Video array of shape (T, H, W, C) with values in [0, 255]
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames could be read from video: {video_path}")

        return np.stack(frames, axis=0)  # Shape: (T, H, W, C)

    def check_video_dimensions(self, video1: np.ndarray, video2: np.ndarray,
                              path1: str = "", path2: str = "") -> None:
        """
        Check if two videos have the same dimensions.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)
            path1: Path to first video (for error messages)
            path2: Path to second video (for error messages)

        Raises:
            ValueError: If dimensions don't match
        """
        if video1.shape != video2.shape:
            raise ValueError(
                f"Video dimensions do not match!\n"
                f"Video 1 {path1}: {video1.shape} (T, H, W, C)\n"
                f"Video 2 {path2}: {video2.shape} (T, H, W, C)"
            )

    def compute_mse(self, video1: np.ndarray, video2: np.ndarray) -> float:
        """
        Compute Mean Squared Error between two videos.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)

        Returns:
            float: MSE value
        """
        return float(np.mean((video1.astype(np.float32) - video2.astype(np.float32)) ** 2))

    def compute_ssim(self, video1: np.ndarray, video2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two videos.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)

        Returns:
            float: Average SSIM value across all frames
        """
        if not SSIM_AVAILABLE:
            raise ImportError("scikit-image not available. Install with: pip install scikit-image")

        ssim_values = []
        for i in range(video1.shape[0]):
            frame1 = video1[i]
            frame2 = video2[i]

            # Compute SSIM for each frame
            ssim_val = ssim(frame1, frame2, channel_axis=2, data_range=255)
            ssim_values.append(ssim_val)

        return float(np.mean(ssim_values))

    def compute_lpips(self, video1: np.ndarray, video2: np.ndarray) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) between two videos.

        Args:
            video1: First video array (T, H, W, C) in [0, 255]
            video2: Second video array (T, H, W, C) in [0, 255]

        Returns:
            float: Average LPIPS value across all frames
        """
        model = self._load_lpips_model()

        lpips_values = []
        with torch.no_grad():
            for i in range(video1.shape[0]):
                # Convert to tensor and normalize to [-1, 1]
                frame1 = torch.from_numpy(video1[i]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                frame2 = torch.from_numpy(video2[i]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

                frame1 = frame1.to(self.device)
                frame2 = frame2.to(self.device)

                # Compute LPIPS
                lpips_val = model(frame1, frame2)
                lpips_values.append(lpips_val.item())

        return float(np.mean(lpips_values))

    def compute_fid(self, video1: np.ndarray, video2: np.ndarray) -> float:
        """
        Compute Fr�chet Inception Distance between two videos.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)

        Returns:
            float: FID value
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available. Install with: pip install scipy")

        # Extract features using a pretrained network (e.g., InceptionV3)
        # This is a simplified placeholder - in practice, use proper feature extraction
        features1 = self._extract_inception_features(video1)
        features2 = self._extract_inception_features(video2)

        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

        # Calculate FID
        fid = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return float(fid)

    def compute_fvd(self, video1: np.ndarray, video2: np.ndarray) -> float:
        """
        Compute Fr�chet Video Distance between two videos using I3D features.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)

        Returns:
            float: FVD value
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not available. Install with: pip install scipy")

        # Extract I3D features
        features1 = self._extract_i3d_features(video1)
        features2 = self._extract_i3d_features(video2)

        # Calculate mean and covariance
        mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

        # Calculate FVD
        fvd = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return float(fvd)

    def _extract_inception_features(self, video: np.ndarray) -> np.ndarray:
        """
        Extract Inception features from video frames.
        This is a placeholder implementation.

        Args:
            video: Video array (T, H, W, C)

        Returns:
            np.ndarray: Feature array of shape (T, feature_dim)
        """
        # Placeholder: In practice, use a pretrained InceptionV3 model
        # For now, return random features
        warnings.warn("Using placeholder Inception features. Implement proper feature extraction.")
        return np.random.randn(video.shape[0], 2048)

    def _extract_i3d_features(self, video: np.ndarray) -> np.ndarray:
        """
        Extract I3D features from video.
        This is a placeholder implementation.

        Args:
            video: Video array (T, H, W, C)

        Returns:
            np.ndarray: Feature array
        """
        # Placeholder: In practice, use a pretrained I3D model
        warnings.warn("Using placeholder I3D features. Implement proper feature extraction.")
        return np.random.randn(1, 400)  # I3D typically outputs 400-dim features

    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                   mu2: np.ndarray, sigma2: np.ndarray,
                                   eps: float = 1e-6) -> float:
        """
        Calculate Fr�chet distance between two Gaussian distributions.

        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small value for numerical stability

        Returns:
            float: Fr�chet distance
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def compute_metrics(self, video1: np.ndarray, video2: np.ndarray,
                       metric_names: List[str]) -> Dict[str, float]:
        """
        Compute multiple metrics between two videos.

        Args:
            video1: First video array (T, H, W, C)
            video2: Second video array (T, H, W, C)
            metric_names: List of metric names to compute

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their values
        """
        supported_metrics = {'mse', 'ssim', 'lpips', 'fid', 'fvd'}
        metric_names_lower = [m.lower() for m in metric_names]

        # Check for unsupported metrics
        unsupported = set(metric_names_lower) - supported_metrics
        if unsupported:
            raise ValueError(f"Unsupported metrics: {unsupported}. Supported: {supported_metrics}")

        results = {}

        for metric_name in metric_names_lower:
            if metric_name == 'mse':
                results['mse'] = self.compute_mse(video1, video2)
            elif metric_name == 'ssim':
                results['ssim'] = self.compute_ssim(video1, video2)
            elif metric_name == 'lpips':
                results['lpips'] = self.compute_lpips(video1, video2)
            elif metric_name == 'fid':
                results['fid'] = self.compute_fid(video1, video2)
            elif metric_name == 'fvd':
                results['fvd'] = self.compute_fvd(video1, video2)

        return results


def compute_video_distance(gt_videos: List[str], pred_videos: List[str],
                          metric_names: List[str],
                          device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[Dict[str, float]]:
    """
    Compute video quality metrics between ground truth and predicted videos.

    Args:
        gt_videos: List of paths to ground truth video files (mp4)
        pred_videos: List of paths to predicted video files (mp4)
        metric_names: List of metric names to compute (mse, ssim, lpips, fid, fvd)
        device: Device to use for computation ('cuda' or 'cpu')

    Returns:
        List[Dict[str, float]]: List of dictionaries containing metrics for each video pair

    Raises:
        ValueError: If input lists have different lengths or video dimensions don't match
    """
    if len(gt_videos) != len(pred_videos):
        raise ValueError(
            f"Number of ground truth videos ({len(gt_videos)}) must match "
            f"number of predicted videos ({len(pred_videos)})"
        )

    if len(gt_videos) == 0:
        raise ValueError("No videos provided")

    # Initialize metric calculator
    metric_calculator = VideoMetric(device=device)

    results = []

    for i, (gt_path, pred_path) in enumerate(zip(gt_videos, pred_videos)):
        print(f"Processing video pair {i+1}/{len(gt_videos)}: {Path(gt_path).name} vs {Path(pred_path).name}")

        # Load videos
        gt_video = metric_calculator.load_video(gt_path)
        pred_video = metric_calculator.load_video(pred_path)

        # Check dimensions
        metric_calculator.check_video_dimensions(gt_video, pred_video, gt_path, pred_path)

        # Compute metrics
        metrics = metric_calculator.compute_metrics(gt_video, pred_video, metric_names)
        results.append(metrics)

        print(f"  Results: {metrics}")

    return results


# if __name__ == "__main__":
#     # Example usage
#     gt_videos = ["path/to/gt_video1.mp4", "path/to/gt_video2.mp4"]
#     pred_videos = ["path/to/pred_video1.mp4", "path/to/pred_video2.mp4"]
#     metric_names = ["mse", "ssim", "lpips"]

#     try:
#         results = compute_video_distance(gt_videos, pred_videos, metric_names)

#         print("\nFinal Results:")
#         for i, result in enumerate(results):
#             print(f"Video pair {i+1}: {result}")
#     except Exception as e:
#         print(f"Error: {e}")
