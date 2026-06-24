module github.com/guygrigsby/lmkit-go/backend

go 1.26

require (
	github.com/gomlx/compute v0.0.0-20260621195442-7cf34e76eacb
	github.com/gomlx/gomlx v0.27.4-0.20260622114827-c34b0fdb10e7
)

require (
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/gofrs/flock v0.13.0 // indirect
	github.com/gomlx/go-xla v0.2.3-0.20260622114419-ab1f50d463f9 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	golang.org/x/exp v0.0.0-20260410095643-746e56fc9e2f // indirect
	golang.org/x/sys v0.43.0 // indirect
	golang.org/x/term v0.40.0 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
	k8s.io/klog/v2 v2.140.0 // indirect
)

replace github.com/gomlx/gomlx => github.com/guygrigsby/gomlx v0.27.4-0.20260624142714-2f1b83e85141

replace github.com/gomlx/compute => github.com/guygrigsby/compute v0.0.0-20260623170013-d291a824cc40

replace github.com/gomlx/go-xla => github.com/guygrigsby/go-xla v0.2.3-0.20260622220527-d2d893bf5dbc
