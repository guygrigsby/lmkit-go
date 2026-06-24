package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train/loss"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// TestTiedLogitsCrossEntropyChunkedMatchesFull pins that the chunked tied-logits cross
// entropy equals the full-logits SparseCategoricalCrossEntropyLogits, in both the scalar
// loss and the gradients w.r.t. h and the table, for chunk sizes that do and don't divide
// the vocab (and the degenerate single-chunk case).
func TestTiedLogitsCrossEntropyChunkedMatchesFull(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	const B, T, H, V = 2, 4, 8, 20

	hData := rampSeed(B*T*H, 3)
	tableData := rampSeed(V*H, 9)
	labelData := make([]int32, B*T) // a spread of target ids in [0,V)
	for i := range labelData {
		labelData[i] = int32((i * 7) % V)
	}

	// build returns (loss, d loss/d h, d loss/d table) flattened, for the given loss closure.
	// Same constant inputs every call so results are comparable.
	build := func(lossFn func(gr *g.Graph, h, table, labels *g.Node) *g.Node) (lossV, dH, dTable []float32) {
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) []*g.Node {
			h := g.Const(gr, tensors.FromFlatDataAndDimensions(hData, B, T, H))
			table := g.Const(gr, tensors.FromFlatDataAndDimensions(tableData, V, H))
			labels := g.Const(gr, tensors.FromFlatDataAndDimensions(labelData, B, T, 1))
			lo := lossFn(gr, h, table, labels)
			grads := g.Gradient(lo, h, table)
			return []*g.Node{lo, grads[0], grads[1]}
		})
		res := exec.MustCall()
		return tensors.MustCopyFlatData[float32](res[0]),
			tensors.MustCopyFlatData[float32](res[1]),
			tensors.MustCopyFlatData[float32](res[2])
	}

	fullLoss, fullDH, fullDT := build(func(gr *g.Graph, h, table, labels *g.Node) *g.Node {
		logits := model.TiedLogits(h, table) // [B,T,V]
		return loss.SparseCategoricalCrossEntropyLogits([]*g.Node{labels}, []*g.Node{logits})
	})

	for _, chunk := range []int{V, 7, 5} { // single chunk; 7 doesn't divide 20; 5 divides
		cLoss, cDH, cDT := build(func(gr *g.Graph, h, table, labels *g.Node) *g.Node {
			return model.TiedLogitsCrossEntropyChunked(h, table, labels, chunk)
		})
		assertClose(t, "loss chunk="+itoa(chunk), fullLoss, cLoss, 1e-4)
		assertClose(t, "dH chunk="+itoa(chunk), fullDH, cDH, 1e-4)
		assertClose(t, "dTable chunk="+itoa(chunk), fullDT, cDT, 1e-4)
	}
}
