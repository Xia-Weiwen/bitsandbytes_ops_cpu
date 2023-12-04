import torch

def is_on_cpu(tensors):
    on_cpu = True
    for t in tensors:
        if t is None: continue # NULL pointers are fine
        on_cpu &= (t.device.type == 'cpu')
    if not on_cpu:
        raise TypeError(
            'All input tensors need to be on CPU, but found some tensors to not be on CPU:\n' \
            f' {[(t.shape, t.device) if isinstance(t, torch.Tensor) else None for t in tensors]}'
        )
    return on_cpu

@torch.compile
def double_quant(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    is_on_cpu([A, col_stats, row_stats, out_col, out_row])
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        assert A.dim() == 2, f"double_quant: Input tensor should be 2d or 3d but got {A.dim()}d"
        rows = A.shape[0]
    A = A.reshape(rows, cols)

    coo_tensor = None

    def get_row_col_stats(A):
        row_stats = torch.max(torch.abs(A), 1).values # absolute max of each row
        col_stats = torch.max(torch.abs(A), 0).values # absolute max of each col
        return row_stats, col_stats
    
    def quant_to_int8(A, stats):
        return torch.clamp(torch.round(A / stats * 127).to(torch.int8), -128, 127)

    if threshold == 0.0:
        if row_stats is None or col_stats is None:
            row_stats, col_stats = get_row_col_stats(A)
    else:
        outlier_indices = torch.abs(A) > threshold # find outliers
        outlier_coord = outlier_indices.nonzero() # get outlier coordinates
        outlier_rows = outlier_coord[:, 0].tolist() # outlier row for COO sparse tensor
        outlier_cols = outlier_coord[:, 1].tolist() # outlier column for COO sparse tensor
        outlier_values = A[outlier_indices].tolist() # outlier values for COO sparse tensor
        coo_tensor = (outlier_rows, outlier_cols, outlier_values)
        A[outlier_indices] = 0 # zero out outliers
        if row_stats is None or col_stats is None:
            row_stats, col_stats = get_row_col_stats(A)

    # quant_by_row = torch.clamp(torch.round(A / row_stats.unsqueeze(-1) * 127, ), -128, 127)
    # quant_by_col = torch.clamp(torch.round(A / col_stats.unsqueeze(0) * 127), -128, 127)
    quant_by_row = quant_to_int8(A, row_stats.unsqueeze(-1))
    quant_by_col = quant_to_int8(A, col_stats.unsqueeze(0))
    if out_row is not None:
        out_row.copy_(quant_by_row)
    else:
        out_row = quant_by_row
    if out_col is not None:
        out_col.copy_(quant_by_col)
    else:
        out_col = quant_by_col
    return out_row, out_col, row_stats, col_stats, coo_tensor
