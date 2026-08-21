"""
Evaluation metrics
Corresponds to paper Results: accuracy, Pearson r, MAE, RMSE, ICC(A,1), 
confusion matrix, calibration analysis.
"""
import numpy as np
import torch
from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error


class SkillMetrics:
    """Skill assessment metrics."""

    @staticmethod
    def accuracy(y_true, y_pred):
        """Classification accuracy."""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def pearson_r(y_true, y_pred):
        """Pearson correlation coefficient."""
        if len(y_true) < 2:
            return 0.0, 1.0
        r, p = pearsonr(y_true, y_pred)
        return r, p

    @staticmethod
    def mae(y_true, y_pred):
        """Mean absolute error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true, y_pred):
        """Root mean squared error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def icc_a1(data, confidence=0.95):
        """
        ICC(A,1) - Two-way random-effects, absolute agreement, single rater.
        Following McGraw & Wong (1996) and Koo & Li (2016).

        Args:
            data: [n_subjects, n_raters] array of ratings.
        Returns:
            icc: ICC(A,1) value
            ci_lower, ci_upper: confidence interval bounds
        """
        data = np.array(data, dtype=float)
        n, k = data.shape

        if n < 2 or k < 2:
            return 0.0, 0.0, 0.0

        grand_mean = np.mean(data)
        row_means = np.mean(data, axis=1)
        col_means = np.mean(data, axis=0)

        # Mean Squares
        SSR = k * np.sum((row_means - grand_mean) ** 2)
        SSC = n * np.sum((col_means - grand_mean) ** 2)
        SSE = np.sum((data - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)

        dfR = n - 1
        dfC = k - 1
        dfE = (n - 1) * (k - 1)

        MSR = SSR / dfR
        MSC = SSC / dfC
        MSE = SSE / dfE

        # ICC(A,1) absolute agreement
        num = MSR - MSE
        den = MSR + (k - 1) * MSE + k * (MSC - MSE) / n

        if den <= 0:
            return 0.0, 0.0, 0.0

        icc = num / den

        # Confidence interval via Fisher z-transformation approximation
        # Simplified CI based on variance components
        var_icc = (2 * (1 - icc) ** 2) / (n - 1)  # approximate
        se = np.sqrt(var_icc)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        z_icc = 0.5 * np.log((1 + icc) / (1 - icc)) if abs(icc) < 1 else 0
        ci_lower = np.tanh(z_icc - z * se)
        ci_upper = np.tanh(z_icc + z * se)

        return float(icc), float(ci_lower), float(ci_upper)

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """Confusion matrix."""
        return confusion_matrix(y_true, y_pred, labels=labels)

    @staticmethod
    def adjacent_grade_error_rate(y_true, y_pred):
        """
        Adjacent-grade vs cross-grade error analysis.
        Paper key metric: errors should only occur between adjacent grades.
        """
        errors = np.abs(y_true - y_pred)
        adjacent = np.sum(errors == 1)
        cross_grade = np.sum(errors >= 2)
        total_errors = np.sum(errors > 0)

        return {
            "adjacent": int(adjacent),
            "cross_grade": int(cross_grade),
            "total_errors": int(total_errors),
            "adjacent_ratio": adjacent / total_errors if total_errors > 0 else 0.0
        }

    @staticmethod
    def calibration_error(y_true, y_prob, n_bins=5):
        """
        Expected Calibration Error (ECE).
        Paper Fig. 4c: reliability diagram.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)

        ece = 0.0
        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > lower) & (confidences <= upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    @classmethod
    def compute_all(cls, y_true_cls, y_pred_cls, y_true_reg, y_pred_reg, y_prob=None):
        """
        Compute all metrics.
        Args:
            y_true_cls: [N] true class labels
            y_pred_cls: [N] predicted class labels
            y_true_reg: [N] true GRS total scores
            y_pred_reg: [N] predicted GRS total scores
            y_prob: [N, 3] softmax probabilities (optional, for calibration)
        """
        results = {
            "accuracy": cls.accuracy(y_true_cls, y_pred_cls),
            "pearson_r": cls.pearson_r(y_true_reg, y_pred_reg)[0],
            "mae": cls.mae(y_true_reg, y_pred_reg),
            "rmse": cls.rmse(y_true_reg, y_pred_reg),
            "confusion_matrix": cls.confusion_matrix(y_true_cls, y_pred_cls, labels=[0,1,2]).tolist(),
            "error_structure": cls.adjacent_grade_error_rate(y_true_cls, y_pred_cls)
        }

        if y_prob is not None:
            results["calibration_error"] = cls.calibration_error(y_true_cls, y_prob)

        return results
