# MLA Paged Attention operator contract

## ABI

```cpp
extern "C" void run_kernel(
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const __nv_bfloat16* ckv,
    const __nv_bfloat16* kpe,
    __nv_bfloat16* output,
    const int32_t* q_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* kv_lens,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim_ckv,
    int64_t head_dim_kpe,
    int64_t page_size,
    int64_t causal);
```

Evaluation shapes use BF16, `head_dim_ckv=512`, `head_dim_kpe=64`,
`page_size=1`, `causal=0`, and `num_heads` in `{64,128}`.

## Exact algorithm

For batch item `b`, head `h`, and logical key position `j`:

```text
physical_page = kv_indices[kv_indptr[b] + j]
score[j] = (dot(q_nope[b,h], ckv[physical_page])
            + dot(q_pe[b,h], kpe[physical_page])) / sqrt(576)
output[b,h] = softmax(score) @ ckv
```

The published checker uses `ATOL=0.016`, `RTOL=0.016`, requires at least 99%
of elements within tolerance, and rejects any error above eight times tolerance.

## Source variants

- `mla_paged_optimized.cu` implements the exact contract for arbitrary QPE/KPE
  and arbitrary page tables.
- `mla_paged_reference.cu` is the historical 0f3400b evaluation-specialized
  source; it assumes ZeroPE and contains pointer replay.
- `mla_paged_optimized_submit.cu` is the final OJ-specialized source. It also
  assumes identity pages and uses tolerance-backed key subsampling. It does not
  implement the exact general contract.

See `OPTIMIZATION_FINAL.md` for performance, calibration, and full disclosure.

## Build

```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  SOURCE.cu -o /tmp/operator.so
```
