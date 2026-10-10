# Three-tier SimpleCPUOffloadConnector emulator

This experimental overlay runs a modified vLLM image with one GPU cache and
two independent pinned-host KV-cache pools. It is intended for research on
future secondary-memory placement policies. It is not part of the supported
vLLM native `OffloadingConnector` path.

The checked-in image name is a placeholder:

```text
localhost/kv-cache-offload-vllm:dev
```

Build the image from the modified vLLM checkout, push it to a registry visible
to the cluster, then replace the image in `cpu/base/patch-vllm.yaml` or apply an
outer Kustomize image override.

The default configuration allocates 8 GiB to the fast host pool and 32 GiB to
the slow host pool. Requests with vLLM priority below `1` use the fast pool.
Both transfer-time limits are disabled by default; set measured bandwidth and
latency values before using the overlay for emulation results.

The pod requests 96 GiB and is limited to 128 GiB. Recalculate that budget for
the model, tensor-parallel world size, both pinned pools, and runtime overhead.

Render before applying:

```bash
kubectl kustomize guides/tiered-prefix-cache/modelserver/gpu/vllm/emulator/cpu/base
```

Deploy only after replacing the image and checking the rendered command:

```bash
kubectl apply -n "${NAMESPACE}" -k \
  guides/tiered-prefix-cache/modelserver/gpu/vllm/emulator/cpu/base
```

The overlay enables vLLM's development cache endpoint. A cold-cache run can
POST `/reset_prefix_cache?reset_external=true`; the response body is
`{"success":true}` only after the GPU cache and both host pools are clear.
Reset is refused while requests or asynchronous transfers still own cache
state, so callers should retry rather than treating HTTP 200 alone as success.
