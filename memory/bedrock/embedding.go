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

// Package bedrock provides EmbeddingModel implementations for AWS Bedrock models.
package bedrock

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const (
	DefaultModelID   = "amazon.titan-embed-text-v2:0"
	DefaultDimension = 1024
)

// BedrockRuntimeClient abstracts the AWS Bedrock Runtime client for testability.
type BedrockRuntimeClient interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

// Config holds configuration for TitanEmbedding.
type Config struct {
	// Region is the AWS region (e.g., "us-east-1").
	Region string
	// ModelID is the Bedrock model ID. Defaults to "amazon.titan-embed-text-v2:0" if empty.
	ModelID string
	// Dimension is the embedding vector dimension (256, 512, or 1024). Defaults to 1024 if 0.
	Dimension int

	// Client allows injecting a custom BedrockRuntimeClient (for testing).
	// If nil, a real bedrockruntime.Client is created using config.LoadDefaultConfig.
	Client BedrockRuntimeClient
}

// TitanEmbedding implements the postgres.EmbeddingModel interface using
// AWS Bedrock's Amazon Titan Embeddings V2. Authentication uses the standard
// AWS credential chain (environment variables, IAM roles, profiles, etc.).
type TitanEmbedding struct {
	client  BedrockRuntimeClient
	modelID string
	dim     int
}

// embedRequest is the JSON request body for the Titan Embeddings V2 InvokeModel API.
type embedRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int    `json:"dimensions"`
	Normalize  bool   `json:"normalize"`
}

// embedResponse is the JSON response body from the Titan Embeddings V2 InvokeModel API.
type embedResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

// New creates a new TitanEmbedding. Returns an error if AWS SDK configuration loading fails.
func New(ctx context.Context, cfg Config) (*TitanEmbedding, error) {
	dim := cfg.Dimension
	if dim == 0 {
		dim = DefaultDimension
	}

	modelID := cfg.ModelID
	if modelID == "" {
		modelID = DefaultModelID
	}

	client := cfg.Client
	if client == nil {
		var opts []func(*config.LoadOptions) error
		if cfg.Region != "" {
			opts = append(opts, config.WithRegion(cfg.Region))
		}

		awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
		if err != nil {
			return nil, fmt.Errorf("failed to load AWS config: %w", err)
		}

		client = bedrockruntime.NewFromConfig(awsCfg)
	}

	return &TitanEmbedding{
		client:  client,
		modelID: modelID,
		dim:     dim,
	}, nil
}

// Dimension returns the configured embedding vector dimension.
func (e *TitanEmbedding) Dimension() int {
	return e.dim
}

// Embed generates an embedding vector for the given text using AWS Bedrock.
func (e *TitanEmbedding) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := embedRequest{
		InputText:  text,
		Dimensions: e.dim,
		Normalize:  true,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal bedrock request: %w", err)
	}

	resp, err := e.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(e.modelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        bodyBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("bedrock InvokeModel failed: %w", err)
	}

	var embedResp embedResponse
	if err := json.Unmarshal(resp.Body, &embedResp); err != nil {
		return nil, fmt.Errorf("failed to decode bedrock response: %w", err)
	}

	if len(embedResp.Embedding) == 0 {
		return nil, fmt.Errorf("bedrock returned empty embedding")
	}

	return embedResp.Embedding, nil
}
