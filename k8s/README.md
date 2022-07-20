# [Homework 4] Kubernetes Cluster

# Run
1. Enable kubernetes in Docker Desktop 
2. Create new namespace `kubectl create ns test`
3. Apply any manifest `kubectl apply -n test -f online-inference-deployment-rolling-update.yaml`

Extra:
- Get Pods `kubectl -n test get pods`
- Get Pod description `kubectl describe -n test pod heart-desease-server` 
