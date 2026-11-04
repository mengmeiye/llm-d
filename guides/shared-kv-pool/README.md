# Shared KV-Cache Pool

## Overview

By default every vLLM replica owns its KV cache. Replicas that see the same
prefixes each compute their own copy, a failed replica's cache is lost with
it, and a replica added by the autoscaler starts cold. This guide makes the
KV cache a property of the node instead of the replica: the replicas on a
GPU node share one KV pool in node memory (tmpfs), so

- a prefix computed by one replica is loaded, not recomputed, by its siblings;
- requests a deleted or failed replica would have served route to a sibling
  that loads the failed replica's blocks from the pool;
- a replica added to the node finds the blocks its siblings already wrote,
  and the router routes to it as warm before it has served a request.

Three pieces make this work:

| Piece | Role |
| --- | --- |
| vLLM `OffloadingConnector` with a `TieringOffloadingSpec` | Each replica keeps a private CPU tier and an `fs` secondary tier on the node pool. Block files are named by content hash and published atomically, so replicas share blocks without coordination. The tier attaches to an exact pool generation (`pool_generation`) and refreshes block access times for eviction. |
| `kv-pool-manager` DaemonSet | Owns each node's pool: publishes a complete generation before any replica attaches, keeps it within eviction watermarks (least recently used blocks first), cleans up after crashed writers, and destroys a generation only when no replica holds it and the caller names its generation. Pools outlive the manager and every replica. |
| llm-d router, `precise-prefix-cache-producer` | Marks the `storage` tier `shared: node`, so blocks in the pool are indexed under the node, not under the replica that wrote them. |

The pool lives in node memory, so it does not survive a node reboot, and a
replica on another node cannot read it. For cross-node reuse, put the pool on
a `ReadWriteMany` volume and mark the tier `shared: pool` (see
[Tuning](#tuning)).

## Supported Accelerators and Model Servers

<!-- guide:support start -->
| Accelerator | `ACCELERATOR_TYPE` | Served model | vLLM | Notes |
| --- | --- | --- | --- | --- |
| NVIDIA GPU | `gpu` | `Qwen/Qwen3-8B` | 🟡 community | Default. Any NVIDIA GPU that serves Qwen3-8B · 2 replicas × 1 GPU on one node · `INFRA_PROVIDER`: `base`, `ocp` |

✅ validated: covered by a nightly E2E workflow · 🟡 community: maintained by the hardware vendor or community, not covered by nightly E2E · ❌ not supported: tracked in the linked issue · — no configuration.
<!-- guide:support end -->

## Prerequisites

- Have the [proper client tools installed on your local system](../../helpers/client-setup/README.md) to use this guide.
- Create a [HuggingFace token](../../helpers/hf-token.md) and export it as `HF_TOKEN` in your shell.
- A vLLM image with managed `fs` pools (`pool_generation`), and an llm-d router image with shared tiers (`kvCacheBackendConfigs[].shared`). Set them in `modelserver/gpu/vllm/base/kustomization.yaml` and `router/shared-kv-pool.values.yaml`.
- Node memory for the pool (`KV_POOL_CAPACITY_BYTES`, 64 GiB by default) in addition to each replica's CPU tier. `/dev/shm` on the node must be large enough to hold it.
- (Optional) Install the [monitoring stack](../../docs/operations/observability/setup.md) if you plan to enable Prometheus monitoring.

### Get the guide

<!-- guide:prerequisites.clone start -->
<!-- llm-d-cicd:skip start -->
```bash
export BRANCH=main
git clone https://github.com/llm-d/llm-d.git && cd llm-d && git checkout ${BRANCH}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.clone end -->

### Configure the environment

<!-- guide:env.static start -->
```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
export GUIDE_NAME=shared-kv-pool
export NAMESPACE=llm-d-shared-kv-pool
export MONITORING=false # options: false, true
export MONITORING_VALUES=
export ACCELERATOR_TYPE=gpu # options: gpu
export MODEL_SERVER=vllm # options: vllm
export INFRA_PROVIDER=base # options: base, ocp
export MODEL=Qwen/Qwen3-8B
export KV_POOL_NAME=qwen3-8b
source ${REPO_ROOT}/guides/env.sh # defines GAIE_VERSION, ROUTER_CHART_VERSION, router chart URLs, and CURL_TEST_IMAGE
```
<!-- guide:env.static end -->

Install the Gateway API Inference Extension CRDs:

<!-- guide:prerequisites.gaie start -->
```bash
# GAIE_URL is automatically calculated from GAIE_VERSION at ${REPO_ROOT}/guides/env.sh
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/${GAIE_URL}/v1-manifests.yaml
```
<!-- guide:prerequisites.gaie end -->

Create a target namespace for the installation:

<!-- guide:prerequisites.namespace start -->
```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
```
<!-- guide:prerequisites.namespace end -->

Create the `llm-d-hf-token` secret in your target namespace:

<!-- guide:prerequisites.secrets start -->
<!-- llm-d-cicd:skip start -->
```bash
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.secrets end -->

On OpenShift, allow the pool's hostPath volume:

<!-- guide:prerequisites.openshift_scc start -->
<!-- llm-d-cicd:skip start -->
```bash
# only when INFRA_PROVIDER=ocp:
# The pool lives on a hostPath. Let the pool manager and the model
# servers mount it (cluster-admin).
oc adm policy add-scc-to-user hostmount-anyuid -z default -n ${NAMESPACE}
oc adm policy add-scc-to-user hostmount-anyuid -z ${GUIDE_NAME}-gpu-vllm-sa -n ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:prerequisites.openshift_scc end -->

## Installation Instructions

### 1. Deploy the Pool Manager

<!-- guide:deploy.pool_manager start -->
```bash
# One pool manager per GPU node publishes the node's pool before any
# model server attaches to it.
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/pool-manager/
kubectl rollout status -n ${NAMESPACE} daemonset/${GUIDE_NAME}-kv-pool-manager --timeout=300s
```
<!-- guide:deploy.pool_manager end -->

### 2. Deploy the llm-d Router

<!-- guide:deploy.router_values start -->
```bash
export ROUTER_BASE_VALUES="${REPO_ROOT}/guides/recipes/router/base.values.yaml"
export ROUTER_VALUES="${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml"
```
<!-- guide:deploy.router_values end -->

(Optional) Enable Prometheus monitoring on the `llm-d` router:

<!-- guide:deploy.monitoring_values start -->
```bash
# only when MONITORING=true:
export MONITORING_VALUES="-f ${REPO_ROOT}/guides/recipes/router/features/monitoring.values.yaml"
```
<!-- guide:deploy.monitoring_values end -->

<!-- guide:deploy.standalone start -->
```bash
helm install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${ROUTER_BASE_VALUES} \
  ${MONITORING_VALUES} \
  -f ${ROUTER_VALUES} \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```
<!-- guide:deploy.standalone end -->

### 3. Deploy the Model Servers

Each replica waits until its node's pool is published, then attaches to that
generation:

<!-- guide:deploy.modelserver start -->
```bash
kubectl apply -n ${NAMESPACE} \
  -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${INFRA_PROVIDER}/
```
<!-- guide:deploy.modelserver end -->

(Optional) Deploy the monitoring resources for model servers:

<!-- guide:deploy.monitoring start -->
```bash
# only when MONITORING=true:
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring
```
<!-- guide:deploy.monitoring end -->

## Verification

### 1. Get the IP of the Proxy

<!-- guide:verify.endpoint.standalone start -->
```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```
<!-- guide:verify.endpoint.standalone end -->

### 2. Check the Pools

`attached` is `true` while at least one model server holds the generation:

<!-- guide:verify.tests.pool start -->
```bash
# Each node's pool is published, READY, and attached by its model servers
for POOL_POD in $(kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=kv-pool-manager -o jsonpath='{.items[*].metadata.name}'); do
  kubectl exec -n ${NAMESPACE} ${POOL_POD} -- \
    python3 /opt/kv-pool-manager/kv_pool_manager.py status --name ${KV_POOL_NAME}
done
```
<!-- guide:verify.tests.pool end -->

### 3. Send a Test Request

<!-- guide:verify.tests.request start -->
```bash
kubectl run curl-test --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="IP=${IP}" \
  --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'curl -sS -X POST "http://${IP}/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"How are you today?\"}"'
```
<!-- guide:verify.tests.request end -->

### 4. Verify Cross-Replica Reuse

<!-- guide:verify.tests.cross_replica start -->
```bash
# Prime a ~4k-token prefix on one replica, then send the same prompt
# straight to the other one: it loads the blocks from the pool
# instead of recomputing them (external prefix-cache hits rise).
# Two model servers on one node, as "<podIP> <podIP>"
read A B <<<"$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/role=decode \
  -o jsonpath='{range .items[*]}{.status.hostIP} {.status.podIP}{"\n"}{end}' \
  | sort | awk '$1 == h {print p, $2; exit} {h = $1; p = $2}')"
test -n "${B}" || { echo "FAIL: no two model servers share a node"; exit 1; }
kubectl run pool-check --rm -i --restart=Never \
  --image=${CURL_TEST_IMAGE} \
  --namespace="${NAMESPACE}" \
  --env="A=${A}" --env="B=${B}" --env="MODEL=${MODEL}" \
  -- /bin/sh -c 'P=$(for i in $(seq 1 200); do printf "A shared KV pool lets every replica on a node reuse a prefix. "; done)
    hits() { curl -sS "http://$1:8000/metrics" | awk "/^vllm:external_prefix_cache_hits_total/ {s += \$NF} END {print s + 0}"; }
    curl -sS -o /dev/null -X POST "http://${A}:8000/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"${P}\", \"max_tokens\": 1}"
    sleep 2
    BEFORE=$(hits ${B})
    curl -sS -o /dev/null -X POST "http://${B}:8000/v1/completions" -H "Content-Type: application/json" -d "{\"model\": \"${MODEL}\", \"prompt\": \"${P}\", \"max_tokens\": 1}"
    AFTER=$(hits ${B})
    echo "replica B external prefix-cache hits: ${BEFORE} -> ${AFTER}"
    test "${AFTER}" != "${BEFORE}" || { echo "FAIL: replica B did not load the prefix from the pool"; exit 1; }'
```
<!-- guide:verify.tests.cross_replica end -->

## Cleanup

<!-- guide:cleanup.modelserver start -->
```bash
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${INFRA_PROVIDER}
```
<!-- guide:cleanup.modelserver end -->

<!-- guide:cleanup.pool start -->
```bash
# Destroy each node's pool generation, then the managers. Destroy
# refuses (exit code 3) while a model server is still attached, so
# wait for the model servers to terminate first.
kubectl wait -n ${NAMESPACE} --for=delete pod -l llm-d.ai/guide=${GUIDE_NAME},llm-d.ai/role=decode --timeout=180s
for POOL_POD in $(kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/component=kv-pool-manager -o jsonpath='{.items[*].metadata.name}'); do
  GENERATION=$(kubectl exec -n ${NAMESPACE} ${POOL_POD} -- cat /kv-pool/.${KV_POOL_NAME}.ready)
  kubectl exec -n ${NAMESPACE} ${POOL_POD} -- \
    python3 /opt/kv-pool-manager/kv_pool_manager.py destroy --name ${KV_POOL_NAME} --generation ${GENERATION}
done
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/pool-manager/
```
<!-- guide:cleanup.pool end -->

<!-- guide:cleanup.rest start -->
```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}

kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/recipes/modelserver/components/monitoring --ignore-not-found=true
```
<!-- llm-d-cicd:skip start -->
```bash
kubectl delete namespace ${NAMESPACE}
```
<!-- llm-d-cicd:skip end -->
<!-- guide:cleanup.rest end -->

## Troubleshooting

- **A model server stays in `Init`.** Its node has no published pool yet. Check `kubectl logs -n ${NAMESPACE} -l app.kubernetes.io/component=kv-pool-manager`.
- **vLLM fails with `expected <generation>`.** The pool was destroyed and recreated while the pod was starting. Delete the pod so its init container reads the new generation.
- **`destroy` exits with code 3.** A model server still holds the generation. Wait for it to terminate; never remove the pool directory by hand while one is attached.
- **No external prefix-cache hits on the second replica.** Every replica sharing a pool must use the same model, `--block-size`, dtype, `--prefix-caching-hash-algo`, and `PYTHONHASHSEED`.

## Tuning

- **Pool capacity.** `KV_POOL_CAPACITY_BYTES` with `KV_POOL_HIGH_WATERMARK`/`KV_POOL_LOW_WATERMARK` in `pool-manager/daemonset.yaml`. Eviction removes the least recently used block files; a replica that loses the race for a block recomputes it.
- **Router tier weight.** The `storage` weight (0.6) ranks a pool hit below a replica's own GPU (1.0) and CPU (0.8) hits, since a pool hit is copied into the replica's CPU tier before the GPU load.
- **Cross-node pool.** To survive node loss, point the `fs` tier's `root_dir` at a pool on a `ReadWriteMany` volume instead of the node's tmpfs, run one pool manager for it (a Deployment with one replica instead of the DaemonSet), and set `shared: pool` on the router's `storage` tier. The router keys shared scopes by the event's medium, and every `fs` tier reports `STORAGE`, so a deployment uses either node-scoped or pool-scoped sharing, not both.
- **Replica placement.** The model servers prefer to pack onto one node so more replicas share a pool. Use `requiredDuringScheduling` affinity to force it.
