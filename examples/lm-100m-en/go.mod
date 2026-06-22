module github.com/guygrigsby/lmkit-go/examples/lm-100m-en

go 1.26

require (
	github.com/guygrigsby/lmkit-go/data v0.0.0-00010101000000-000000000000
	github.com/guygrigsby/lmkit-go/model v0.0.0-00010101000000-000000000000
	github.com/guygrigsby/lmkit-go/train v0.0.0-00010101000000-000000000000
)

require (
	github.com/NVIDIA/go-nvml v0.13.2-0 // indirect
	github.com/edsrzf/mmap-go v1.2.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/compute v0.0.0-20260621195442-7cf34e76eacb // indirect
	github.com/gomlx/go-xla v0.2.3-0.20260622114419-ab1f50d463f9 // indirect
	github.com/gomlx/gomlx v0.27.4-0.20260622114827-c34b0fdb10e7 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/guygrigsby/lmkit-go/backend v0.0.0-00010101000000-000000000000 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	golang.org/x/exp v0.0.0-20260410095643-746e56fc9e2f // indirect
	golang.org/x/sys v0.43.0 // indirect
	golang.org/x/term v0.40.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	k8s.io/klog/v2 v2.140.0 // indirect
)

replace (
	github.com/guygrigsby/lmkit-go/backend => ../../backend
	github.com/guygrigsby/lmkit-go/data => ../../data
	github.com/guygrigsby/lmkit-go/model => ../../model
	github.com/guygrigsby/lmkit-go/train => ../../train
)
