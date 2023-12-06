import unittest
import torch
import sys
sys.path.append(sys.path[0])
from int8_ops import (
    double_quant,
    transform,
    igemmlt,
    mm_dequant,
    extract_outliers,
)
import itertools


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

    def test_transform(self):
        A = torch.rand(32, 64)
        B, _ = transform(A)
        assert torch.equal(B, A)
        B, _ = transform(A, transpose=True)
        assert torch.equal(B, A.T)

    def test_igemmlt(self):
        shapeA_list = [(32, 64), (2, 32, 64)]
        shapeB = (64, 64)
        for shapeA in shapeA_list:
            A = torch.rand(shapeA)
            A_min, A_max = A.aminmax()
            A_scale = torch.max(A_max, A_min.neg()) / 127
            A_int8 = torch.round(A / A_scale).to(torch.int8)
            B = torch.randn(shapeB)
            B_min, B_max = B.aminmax(dim=1)
            B_scale = torch.max(B_max, B_min.neg()) / 127
            B_int8 = torch.round(B / B_scale.unsqueeze(-1)).to(torch.int8)
            C, _ = igemmlt(A_int8, B_int8)
            C_ref = A_int8.float() @ B_int8.float().T
            C_ref = C_ref.to(torch.int32)
            assert C.dtype == torch.int32
            assert torch.equal(C, C_ref)
            # Test with given out buffer
            C_out = torch.zeros_like(C)
            C, _ = igemmlt(A_int8, B_int8, out=C_out)
            assert id(C) == id(C_out)
            assert torch.equal(C_out, C_ref)

    def test_mm_dequant(self):
        shapeA_list = [(32, 64), (2, 32, 64)]
        use_bias_list = [True, False]
        shapeB = (64, 64)
        cases = itertools.product(shapeA_list, use_bias_list)
        for shapeA, use_bias in cases:
            A = torch.rand(shapeA)
            A_min, A_max = A.aminmax(dim=-1)
            A_stats = torch.max(A_max, A_min.neg())
            A_scale = A_stats / 127
            A_int8 = torch.round(A / A_scale.unsqueeze(-1)).to(torch.int8)
            B = torch.randn(shapeB)
            B_min, B_max = B.aminmax(dim=-1)
            B_stats = torch.max(B_max, B_min.neg())
            B_scale = B_stats / 127
            B_int8 = torch.round(B / B_scale.unsqueeze(-1)).to(torch.int8)
            # Compute dtype is always float after torch.compile
            comp_dtype = torch.float
            out_dtype = mm_dequant.output_dtype
            bias = torch.randn(shapeB[0]).to(comp_dtype) if use_bias else None
            C_i32, _ = igemmlt(A_int8, B_int8)
            C_dq = mm_dequant(C_i32, None, A_stats, B_stats, bias=bias)
            A_scale_for_dq = A_stats.reshape(-1).unsqueeze(-1).to(comp_dtype) / 127
            B_scale_for_dq = B_stats.reshape(-1).unsqueeze(0).to(comp_dtype) / 127
            C_i32_reshaped = C_i32.reshape(-1, C_i32.size(-1))
            C_dq_ref = C_i32_reshaped.to(comp_dtype) * A_scale_for_dq * B_scale_for_dq
            C_dq_ref = C_dq_ref.to(comp_dtype) + (bias if use_bias else 0)
            C_dq_ref = C_dq_ref.to(out_dtype)
            assert torch.allclose(C_dq, C_dq_ref, atol=1e-3, rtol=1e-5)

    def test_extract_outliers(self):
        shapeA = (4096, 4096 * 4)
        idx = torch.unique(torch.randint(0, shapeA[1], size=(10,)).int())
        A = torch.randint(-128, 127, size=shapeA).to(torch.int8)
        outliers_ref = A[:, idx.long()].contiguous()
        outliers = extract_outliers(A, None, idx)
        assert torch.equal(outliers, outliers_ref)


if __name__ == '__main__':
    unittest.main()
