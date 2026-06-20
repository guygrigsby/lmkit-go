package model_test

import (
	"encoding/json"
	"os"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestEmbeddingParity(t *testing.T) {
	// embedding.json has two expected outputs; load the raw shape here.
	raw, err := os.ReadFile("testdata/embedding.json")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var fx struct {
		Inputs         map[string]paritytest.Tensor `json:"inputs"`
		Weights        map[string]paritytest.Tensor `json:"weights"`
		ExpectedEmbed  paritytest.Tensor            `json:"expected_embed"`
		ExpectedLogits paritytest.Tensor            `json:"expected_logits"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("decode: %v", err)
	}
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}

	// ids were serialized as float32; build an int32 tensor for Gather.
	idsF := fx.Inputs["ids"]
	idsI := make([]int32, len(idsF.Data))
	for i, v := range idsF.Data {
		idsI[i] = int32(v)
	}
	idsT := tensors.FromFlatDataAndDimensions(idsI, idsF.Shape...)

	// Lookup.
	execL := g.MustNewExec(be.Compute(), func(table, ids *g.Node) *g.Node {
		return model.EmbedLookup(table, ids)
	})
	gotEmbed := execL.MustCall1(fx.Weights["table"].ToTensor(), idsT)
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](gotEmbed), fx.ExpectedEmbed, 1e-5)

	// Tied logits.
	execT := g.MustNewExec(be.Compute(), func(h, table *g.Node) *g.Node {
		return model.TiedLogits(h, table)
	})
	gotLogits := execT.MustCall1(fx.Inputs["h"].ToTensor(), fx.Weights["table"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](gotLogits), fx.ExpectedLogits, 1e-5)
}
