import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

print(DEVICE)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE:tl.constexpr,
):
    PID = tl.program_id(axis = 0)

    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # HBM -> SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=None) # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    # SRAM -> HBM
    tl.store(output_ptr + offsets, output, mask=mask)



def add(x, y):
    output = torch.empty_like(x)

    assert x.device == DEVICE and y.device == DEVICE

    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 调用内核 grid = n_elements / BLOCK_SIZE
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=1024
    )

    return output

def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    z_tri = add(x, y)
    z_ref = x + y

    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed!")    


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)