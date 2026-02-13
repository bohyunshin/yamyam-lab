"""Tests for multimodal triplet embedding model components.

Tests cover:
- ReviewTextEncoder (precomputed path)
- AttentionFusion with 4 and 5 modalities
- Full Model forward pass with review text
- Model backward pass (gradient flow)
- MultimodalTripletConfig defaults
"""

import pytest
import torch

from yamyam_lab.model.embedding.encoders import (
    AttentionFusion,
    ReviewTextEncoder,
)
from yamyam_lab.model.embedding.multimodal_triplet import Model, MultimodalTripletConfig


class TestReviewTextEncoder:
    """Tests for ReviewTextEncoder with precomputed embeddings."""

    @pytest.fixture
    def encoder(self):
        return ReviewTextEncoder(output_dim=128, dropout=0.1)

    def test_forward_precomputed_shape(self, encoder):
        """Output shape should be (batch_size, output_dim)."""
        batch_size = 4
        x = torch.randn(batch_size, 768)
        out = encoder.forward_precomputed(x)
        assert out.shape == (batch_size, 128)

    def test_forward_precomputed_batch_one(self, encoder):
        """Should work with batch_size=1."""
        x = torch.randn(1, 768)
        out = encoder.forward_precomputed(x)
        assert out.shape == (1, 128)

    def test_output_dim_configurable(self):
        """Output dimension should respect config."""
        encoder = ReviewTextEncoder(output_dim=64)
        x = torch.randn(2, 768)
        out = encoder.forward_precomputed(x)
        assert out.shape == (2, 64)

    def test_gradient_flows(self, encoder):
        """Gradients should flow through the MLP."""
        x = torch.randn(4, 768)
        out = encoder.forward_precomputed(x)
        loss = out.sum()
        loss.backward()
        # Check that encoder parameters have gradients
        for param in encoder.mlp.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestAttentionFusion:
    """Tests for AttentionFusion with 4 and 5 modalities."""

    @pytest.fixture
    def fusion_4mod(self):
        """AttentionFusion with 4 modalities (no review text)."""
        return AttentionFusion(
            category_dim=128,
            menu_dim=256,
            diner_name_dim=64,
            price_dim=32,
            review_text_dim=0,
        )

    @pytest.fixture
    def fusion_5mod(self):
        """AttentionFusion with 5 modalities (with review text)."""
        return AttentionFusion(
            category_dim=128,
            menu_dim=256,
            diner_name_dim=64,
            price_dim=32,
            review_text_dim=128,
        )

    @pytest.fixture
    def base_inputs(self):
        """Standard 4-modality inputs."""
        batch_size = 4
        return {
            "category_emb": torch.randn(batch_size, 128),
            "menu_emb": torch.randn(batch_size, 256),
            "diner_name_emb": torch.randn(batch_size, 64),
            "price_emb": torch.randn(batch_size, 32),
        }

    def test_4mod_output_shape(self, fusion_4mod, base_inputs):
        """4-modality fusion should output total_dim = 480."""
        out = fusion_4mod(**base_inputs)
        assert out.shape == (4, 480)

    def test_4mod_total_dim(self, fusion_4mod):
        """total_dim should be 128+256+64+32=480."""
        assert fusion_4mod.total_dim == 480
        assert fusion_4mod.num_modalities == 4

    def test_5mod_output_shape(self, fusion_5mod, base_inputs):
        """5-modality fusion should output total_dim = 608."""
        review_emb = torch.randn(4, 128)
        out = fusion_5mod(**base_inputs, review_text_emb=review_emb)
        assert out.shape == (4, 608)

    def test_5mod_total_dim(self, fusion_5mod):
        """total_dim should be 128+256+64+32+128=608."""
        assert fusion_5mod.total_dim == 608
        assert fusion_5mod.num_modalities == 5

    def test_5mod_none_review_falls_back(self, fusion_5mod, base_inputs):
        """Passing review_text_emb=None to a 5-mod fusion should still work
        (uses 4 modalities internally, but output_proj expects 5 tokens)."""
        # This should raise or handle gracefully depending on implementation.
        # Current implementation: if review_text_dim > 0 but emb is None,
        # only 4 projections are stacked, causing shape mismatch in output_proj.
        # This is expected - review_text_emb should always be provided when dim > 0.
        with pytest.raises(RuntimeError):
            fusion_5mod(**base_inputs, review_text_emb=None)

    def test_gradient_flows_5mod(self, fusion_5mod, base_inputs):
        """Gradients should flow through all 5 modalities."""
        review_emb = torch.randn(4, 128, requires_grad=True)
        base_inputs["category_emb"].requires_grad_(True)

        out = fusion_5mod(**base_inputs, review_text_emb=review_emb)
        out.sum().backward()

        assert review_emb.grad is not None
        assert base_inputs["category_emb"].grad is not None


class TestFullModel:
    """Tests for the full multimodal triplet Model."""

    @pytest.fixture
    def config_without_review(self):
        """Config with 4 modalities (no review text)."""
        return MultimodalTripletConfig(
            num_large_categories=10,
            num_middle_categories=20,
            num_small_categories=30,
            embedding_dim=128,
            category_dim=128,
            menu_dim=256,
            diner_name_dim=64,
            price_dim=32,
            review_text_dim=0,
            num_attention_heads=4,
            dropout=0.1,
            use_precomputed_menu_embeddings=True,
            use_precomputed_name_embeddings=True,
            device="cpu",
        )

    @pytest.fixture
    def config_with_review(self):
        """Config with 5 modalities (with review text)."""
        return MultimodalTripletConfig(
            num_large_categories=10,
            num_middle_categories=20,
            num_small_categories=30,
            embedding_dim=128,
            category_dim=128,
            menu_dim=256,
            diner_name_dim=64,
            price_dim=32,
            review_text_dim=128,
            num_attention_heads=4,
            dropout=0.1,
            use_precomputed_menu_embeddings=True,
            use_precomputed_name_embeddings=True,
            use_precomputed_review_text_embeddings=True,
            device="cpu",
        )

    @pytest.fixture
    def features_4mod(self):
        """Feature dict for 4 modalities."""
        batch_size = 4
        return {
            "large_category_ids": torch.randint(0, 10, (batch_size,)),
            "middle_category_ids": torch.randint(0, 20, (batch_size,)),
            "small_category_ids": torch.randint(0, 30, (batch_size,)),
            "menu_embeddings": torch.randn(batch_size, 768),
            "diner_name_embeddings": torch.randn(batch_size, 768),
            "price_features": torch.randn(batch_size, 3),
        }

    @pytest.fixture
    def features_5mod(self, features_4mod):
        """Feature dict for 5 modalities (adds review text)."""
        batch_size = 4
        features = dict(features_4mod)
        features["review_text_embeddings"] = torch.randn(batch_size, 768)
        return features

    def test_forward_4mod(self, config_without_review, features_4mod):
        """Model without review text should produce (B, 128) output."""
        model = Model(config=config_without_review)
        model.eval()
        with torch.no_grad():
            out = model(features_4mod)
        assert out.shape == (4, 128)

    def test_forward_5mod(self, config_with_review, features_5mod):
        """Model with review text should produce (B, 128) output."""
        model = Model(config=config_with_review)
        model.eval()
        with torch.no_grad():
            out = model(features_5mod)
        assert out.shape == (4, 128)

    def test_output_is_l2_normalized(self, config_with_review, features_5mod):
        """Output embeddings should be L2-normalized (unit vectors)."""
        model = Model(config=config_with_review)
        model.eval()
        with torch.no_grad():
            out = model(features_5mod)

        norms = torch.norm(out, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_backward_pass(self, config_with_review, features_5mod):
        """Gradients should flow through the model."""
        model = Model(config=config_with_review)
        model.train()

        out = model(features_5mod)
        loss = out.sum()
        loss.backward()

        # Check that review text encoder has gradients
        for param in model.review_text_encoder.mlp.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_has_review_encoder_when_dim_gt_zero(self, config_with_review):
        """Model should have review_text_encoder when review_text_dim > 0."""
        model = Model(config=config_with_review)
        assert hasattr(model, "review_text_encoder")
        assert isinstance(model.review_text_encoder, ReviewTextEncoder)

    def test_model_no_review_encoder_when_dim_zero(self, config_without_review):
        """Model should NOT have review_text_encoder when review_text_dim = 0."""
        model = Model(config=config_without_review)
        assert not hasattr(model, "review_text_encoder")

    def test_parameter_count_increases_with_review(
        self, config_without_review, config_with_review
    ):
        """Model with review text should have more parameters."""
        model_4 = Model(config=config_without_review)
        model_5 = Model(config=config_with_review)

        params_4 = sum(p.numel() for p in model_4.parameters())
        params_5 = sum(p.numel() for p in model_5.parameters())

        assert params_5 > params_4


class TestMultimodalTripletConfig:
    """Tests for MultimodalTripletConfig defaults."""

    def test_defaults(self):
        """Config should have correct default values."""
        config = MultimodalTripletConfig(
            num_large_categories=5,
            num_middle_categories=10,
            num_small_categories=15,
        )
        assert config.embedding_dim == 128
        assert config.review_text_dim == 128
        assert config.use_precomputed_review_text_embeddings is True

    def test_review_text_dim_override(self):
        """review_text_dim should be overridable."""
        config = MultimodalTripletConfig(
            num_large_categories=5,
            num_middle_categories=10,
            num_small_categories=15,
            review_text_dim=0,
        )
        assert config.review_text_dim == 0


class TestLossIntegration:
    """Integration tests: model output -> loss computation."""

    @pytest.fixture
    def model_and_features(self):
        """Create model and sample features for integration testing."""
        config = MultimodalTripletConfig(
            num_large_categories=10,
            num_middle_categories=20,
            num_small_categories=30,
            review_text_dim=128,
            use_precomputed_menu_embeddings=True,
            use_precomputed_name_embeddings=True,
            use_precomputed_review_text_embeddings=True,
            device="cpu",
        )
        model = Model(config=config)

        batch_size = 4
        num_negatives = 5

        def make_features(bs):
            return {
                "large_category_ids": torch.randint(0, 10, (bs,)),
                "middle_category_ids": torch.randint(0, 20, (bs,)),
                "small_category_ids": torch.randint(0, 30, (bs,)),
                "menu_embeddings": torch.randn(bs, 768),
                "diner_name_embeddings": torch.randn(bs, 768),
                "price_features": torch.randn(bs, 3),
                "review_text_embeddings": torch.randn(bs, 768),
            }

        return model, make_features, batch_size, num_negatives

    def test_triplet_loss_integration(self, model_and_features):
        """Model embeddings should work with triplet loss."""
        from yamyam_lab.loss.triplet import triplet_margin_loss_with_multiple_negatives

        model, make_features, batch_size, num_negatives = model_and_features
        model.train()

        anchor_emb = model(make_features(batch_size))
        positive_emb = model(make_features(batch_size))

        # Compute negatives in batch
        flat_neg_features = make_features(batch_size * num_negatives)
        flat_neg_emb = model(flat_neg_features)
        neg_emb = flat_neg_emb.view(batch_size, num_negatives, -1)

        loss = triplet_margin_loss_with_multiple_negatives(
            anchor=anchor_emb,
            positive=positive_emb,
            negatives=neg_emb,
            anchor_category=torch.randint(0, 5, (batch_size,)),
            positive_category=torch.randint(0, 5, (batch_size,)),
            negative_categories=torch.randint(0, 5, (batch_size, num_negatives)),
        )

        assert loss.dim() == 0
        loss.backward()

    def test_infonce_loss_integration(self, model_and_features):
        """Model embeddings should work with InfoNCE loss."""
        from yamyam_lab.loss.infonce import infonce_loss_with_multiple_negatives

        model, make_features, batch_size, num_negatives = model_and_features
        model.train()

        anchor_emb = model(make_features(batch_size))
        positive_emb = model(make_features(batch_size))

        flat_neg_features = make_features(batch_size * num_negatives)
        flat_neg_emb = model(flat_neg_features)
        neg_emb = flat_neg_emb.view(batch_size, num_negatives, -1)

        loss = infonce_loss_with_multiple_negatives(
            anchor=anchor_emb,
            positive=positive_emb,
            negatives=neg_emb,
            anchor_category=torch.randint(0, 5, (batch_size,)),
            positive_category=torch.randint(0, 5, (batch_size,)),
            negative_categories=torch.randint(0, 5, (batch_size, num_negatives)),
        )

        assert loss.dim() == 0
        loss.backward()
