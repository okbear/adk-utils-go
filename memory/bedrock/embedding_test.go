// Copyright 2025 achetronic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// mockClient is a mock implementation of BedrockRuntimeClient for testing.
type mockClient struct {
	invokeFunc func(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

func (m *mockClient) InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	return m.invokeFunc(ctx, params, optFns...)
}

// makeEmbeddingBody builds a mock response body with n float32 values.
func makeEmbeddingBody(n int) []byte {
	embedding := make([]float32, n)
	for i := range embedding {
		embedding[i] = float32(i) * 0.01
	}
	body, _ := json.Marshal(embedResponse{Embedding: embedding})
	return body
}

// newTestEmbedding creates a TitanEmbedding with a mock client, bypassing AWS config.
func newTestEmbedding(client BedrockRuntimeClient, dim int) *TitanEmbedding {
	if dim == 0 {
		dim = DefaultDimension
	}
	return &TitanEmbedding{
		client:  client,
		modelID: DefaultModelID,
		dim:     dim,
	}
}

func TestEmbedSuccess(t *testing.T) {
	const dim = 1024
	mock := &mockClient{
		invokeFunc: func(_ context.Context, _ *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
			return &bedrockruntime.InvokeModelOutput{Body: makeEmbeddingBody(dim)}, nil
		},
	}

	e := newTestEmbedding(mock, dim)
	got, err := e.Embed(t.Context(), "hello world")
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if len(got) != dim {
		t.Errorf("expected %d floats, got %d", dim, len(got))
	}
}

func TestEmbedAPIError(t *testing.T) {
	mock := &mockClient{
		invokeFunc: func(_ context.Context, _ *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
			return nil, errors.New("service unavailable")
		},
	}

	e := newTestEmbedding(mock, 1024)
	_, err := e.Embed(t.Context(), "hello")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestEmbedContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	mock := &mockClient{
		invokeFunc: func(ctx context.Context, _ *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
			return nil, ctx.Err()
		},
	}

	e := newTestEmbedding(mock, 1024)
	_, err := e.Embed(ctx, "hello")
	if err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestEmbedEmptyEmbedding(t *testing.T) {
	mock := &mockClient{
		invokeFunc: func(_ context.Context, _ *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
			body, _ := json.Marshal(embedResponse{Embedding: []float32{}})
			return &bedrockruntime.InvokeModelOutput{Body: body}, nil
		},
	}

	e := newTestEmbedding(mock, 1024)
	_, err := e.Embed(t.Context(), "hello")
	if err == nil {
		t.Fatal("expected error for empty embedding, got nil")
	}
}

func TestDimensionDefault(t *testing.T) {
	ctx := t.Context()
	e, err := New(ctx, Config{
		Client:    &mockClient{},
		Dimension: 0,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Dimension() != DefaultDimension {
		t.Errorf("expected default dimension %d, got %d", DefaultDimension, e.Dimension())
	}
}

func TestDimensionCustom(t *testing.T) {
	ctx := t.Context()
	e, err := New(ctx, Config{
		Client:    &mockClient{},
		Dimension: 512,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.Dimension() != 512 {
		t.Errorf("expected dimension 512, got %d", e.Dimension())
	}
}

func TestDefaultModelID(t *testing.T) {
	ctx := t.Context()
	e, err := New(ctx, Config{
		Client:  &mockClient{},
		ModelID: "",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e.modelID != DefaultModelID {
		t.Errorf("expected modelID %q, got %q", DefaultModelID, e.modelID)
	}
}
