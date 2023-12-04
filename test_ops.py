import unittest
import torch
import sys
sys.path.append(sys.path[0])
from int8_ops import (
    double_quant,
)


class Test8bitOps(unittest.TestCase):
    def test_double_quant(self):
        A = torch.rand(64, 64) * 3.0
        threshold = 3.0
        double_quant(A, threshold=threshold)
        # run again
        out_row, out_col, row_stats, col_stats, coo_tensor = double_quant(A, threshold=threshold)
        assert out_row.dtype == torch.int8 and out_col.dtype == torch.int8
        assert row_stats is not None, col_stats is not None
        assert len(coo_tensor) == 3
        outlier_indices = torch.abs(A) > threshold # find outliers
        outlier_coord = outlier_indices.nonzero() # get outlier coordinates
        outlier_rows = outlier_coord[:, 0].tolist() # outlier row for COO sparse tensor
        outlier_cols = outlier_coord[:, 1].tolist() # outlier column for COO sparse tensor
        outlier_values = A[outlier_indices].tolist() # outlier values for COO sparse tensor
        coo_tensor_ref = (outlier_rows, outlier_cols, outlier_values)
        A[outlier_indices] = 0 # zero out outliers
        row_stats_ref = torch.max(torch.abs(A), 1).values # absolute max of each row
        col_stats_ref = torch.max(torch.abs(A), 0).values # absolute max of each col
        out_row_ref = torch.clamp(torch.round(A / row_stats_ref.unsqueeze(-1) * 127).to(torch.int8), -128, 127)
        out_col_ref = torch.clamp(torch.round(A / col_stats_ref.unsqueeze(0) * 127).to(torch.int8), -128, 127)
        assert torch.equal(out_row, out_row_ref)
        assert torch.equal(out_col, out_col_ref)
        assert torch.equal(row_stats, row_stats_ref)
        assert torch.equal(col_stats, col_stats_ref)
        assert all(coo_tensor[i] == coo_tensor_ref[i] for i in range(3))

    def test_double_quant_zero_threshold(self):
        A = torch.rand(64, 64) * 3.0
        threshold = 0.0
        double_quant(A, threshold=threshold)
        # run again
        out_row, out_col, row_stats, col_stats, coo_tensor = double_quant(A, threshold=threshold)
        assert out_row.dtype == torch.int8 and out_col.dtype == torch.int8
        assert row_stats is not None, col_stats is not None
        assert coo_tensor is None
        row_stats_ref = torch.max(torch.abs(A), 1).values # absolute max of each row
        col_stats_ref = torch.max(torch.abs(A), 0).values # absolute max of each col
        out_row_ref = torch.clamp(torch.round(A / row_stats_ref.unsqueeze(-1) * 127).to(torch.int8), -128, 127)
        out_col_ref = torch.clamp(torch.round(A / col_stats_ref.unsqueeze(0) * 127).to(torch.int8), -128, 127)
        assert torch.equal(out_row, out_row_ref)
        assert torch.equal(out_col, out_col_ref)
        assert torch.equal(row_stats, row_stats_ref)
        assert torch.equal(col_stats, col_stats_ref)

    def test_double_quant_out_buffer(self):
        A = torch.rand(64, 64) * 3.0
        threshold = 3.0
        out_row = torch.ones_like(A)
        out_col = torch.ones_like(A)
        out_row2, out_col2, _, _, _ = double_quant(A, out_row=out_row, out_col=out_col, threshold=threshold)
        assert id(out_row) == id(out_row2)
        assert id(out_col) == id(out_col2)


if __name__ == '__main__':
    unittest.main()