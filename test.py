import time
import unittest

import torch

from main import get_parser
from models.backbone import build_backbone
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from models.position import SinePositionEncoder
from utils.data import nested_tensor_from_tensor_list


class TestModelsInit(unittest.TestCase):
    def setUp(self):
        self.args = get_parser().parse_args()
        self.args.num_classes = 91

    def test_detr_init(self):
        build_detr(self.args)


class TestModelsForward(unittest.TestCase):
    def setUp(self):
        print("")

        self.args = get_parser().parse_args()
        self.args.num_classes = 91

        self.pixel_H = 1024
        self.pixel_W = 1024
        self.feat_H = 32
        self.feat_W = 32

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_backbone_forward(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')
        backbone = build_backbone(self.args).to('cuda')

        t0 = time.time()
        backbone(images)
        t1 = time.time()

        print(f"Memory backbone (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (forward): {(t1-t0)*1e3: .1f} ms")

    def test_sine_position_forward(self):
        features = torch.randn(self.args.batch_size, self.args.feat_dim, self.feat_H, self.feat_W).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        sine_pos_encoder = SinePositionEncoder()

        t0 = time.time()
        sine_pos_encoder(features, feature_masks)
        t1 = time.time()

        print(f"Memory sine pos (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sine pose (forward): {(t1-t0)*1e3: .1f} ms")

    def test_encoder_forward(self):
        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        encoder = build_encoder(self.args).to('cuda')

        t0 = time.time()
        encoder(features, feature_masks, pos_encodings)
        t1 = time.time()

        print(f"Memory encoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time encoder (forward): {(t1-t0)*1e3: .1f} ms")

    def test_sample_decoder_forward(self):
        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        decoder = build_decoder(self.args).to('cuda')

        t0 = time.time()
        decoder(features, feature_masks, pos_encodings)
        t1 = time.time()

        print(f"Memory sample decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (forward): {(t1-t0)*1e3: .1f} ms")

    def test_detr_forward(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')

        self.args.num_encoder_layers = 1
        self.args.num_decoder_layers = 1
        detr = build_detr(self.args).to('cuda')

        t0 = time.time()
        detr(images)
        t1 = time.time()

        print(f"Memory DETR model (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (forward): {(t1-t0)*1e3: .1f} ms")


if __name__ == '__main__':
    unittest.main()
