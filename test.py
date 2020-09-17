import unittest

import torch
from torch.utils._benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.decoder import build_decoder
from models.detr import build_detr
from models.encoder import build_encoder
from models.position import SinePositionEncoder
from utils.data import nested_tensor_from_tensor_list


class TestModelsForwardOnly(unittest.TestCase):
    def setUp(self):
        print("")
        self.globals_dict = {}

        self.args = get_parser().parse_args()
        self.args.num_classes = 91

        self.pixel_H = 1024
        self.pixel_W = 1024
        self.feat_H = 32
        self.feat_W = 32

        self.args.lr_backbone = 1e-5
        self.args.lr_encoder = 1e-4
        self.args.lr_decoder = 1e-4

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_backbone(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')
        backbone = build_backbone(self.args).to('cuda')

        self.globals_dict['backbone'] = backbone
        self.globals_dict['images'] = images

        timer = Timer(stmt='backbone(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory backbone (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (forward): {t: .1f} ms")

    def test_sine_position(self):
        features = torch.randn(self.args.batch_size, self.args.feat_dim, self.feat_H, self.feat_W).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        sine_pos_encoder = SinePositionEncoder()

        self.globals_dict['sine_pos_encoder'] = sine_pos_encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks

        timer = Timer(stmt='sine_pos_encoder(features, feature_masks)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sine pos (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sine pose (forward): {t: .1f} ms")

    def test_encoder(self):
        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        encoder = build_encoder(self.args).to('cuda')

        self.globals_dict['encoder'] = encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='encoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory encoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time encoder (forward): {t: .1f} ms")

    def test_global_decoder(self):
        self.args.decoder_type = 'global'

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        decoder = build_decoder(self.args).to('cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory global decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time global decoder (forward): {t: .1f} ms")

    def test_sample_decoder(self):
        self.args.decoder_type = 'sample'

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        decoder = build_decoder(self.args).to('cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sample decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (forward): {t: .1f} ms")

    def test_detr(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')

        self.args.num_encoder_layers = 1
        self.args.num_decoder_layers = 1
        detr = build_detr(self.args).to('cuda')

        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images

        timer = Timer(stmt='detr(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (forward): {t: .1f} ms")


class TestModelsWithBackward(unittest.TestCase):
    def setUp(self):
        print("")
        self.globals_dict = {}

        self.args = get_parser().parse_args()
        self.args.num_classes = 91

        self.pixel_H = 1024
        self.pixel_W = 1024
        self.feat_H = 32
        self.feat_W = 32

        self.args.lr_backbone = 1e-5
        self.args.lr_encoder = 1e-4
        self.args.lr_decoder = 1e-4

    def tearDown(self):
        torch.cuda.reset_peak_memory_stats()

    def test_backbone(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')
        backbone = build_backbone(self.args).to('cuda')

        self.globals_dict['backbone'] = backbone
        self.globals_dict['images'] = images

        timer = Timer(stmt='backbone(images)[-1].tensors.sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory backbone (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (backward): {t: .1f} ms")

    def test_encoder(self):
        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        encoder = build_encoder(self.args).to('cuda')

        self.globals_dict['encoder'] = encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='encoder(features, feature_masks, pos).sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory encoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time encoder (backward): {t: .1f} ms")

    def test_global_decoder(self):
        self.args.decoder_type = 'global'

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        decoder = build_decoder(self.args).to('cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos)[0].sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory global decoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time global decoder (backward): {t: .1f} ms")

    def test_sample_decoder(self):
        self.args.decoder_type = 'sample'

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        feature_masks = (torch.randn(self.args.batch_size, self.feat_H, self.feat_W) > 0).to('cuda')
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim).to('cuda')
        decoder = build_decoder(self.args).to('cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos)[0].sum().backward()', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sample decoder (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (backward): {t: .1f} ms")

    def test_detr(self):
        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')

        self.args.num_encoder_layers = 1
        self.args.num_decoder_layers = 1
        detr = build_detr(self.args).to('cuda')

        stmt = 'torch.cat([tensor for tensor in detr(images)[0].values()], dim=1).sum().backward()'
        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images

        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (backward): {t: .1f} ms")


class TestLoadingFromOriginalDETR(unittest.TestCase):

    def setUp(self):
        self.args = get_parser().parse_args()
        self.args.num_classes = 91

        original_detr_url = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        self.state_dict = torch.hub.load_state_dict_from_url(original_detr_url)['model']

    def test_backbone(self):
        backbone = build_backbone(self.args)
        backbone.load_from_original_detr(self.state_dict)

    def test_encoder(self):
        encoder = build_encoder(self.args)
        encoder.load_from_original_detr(self.state_dict)

    def test_global_decoder(self):
        self.args.decoder_type = 'global'
        decoder = build_decoder(self.args)
        decoder.load_from_original_detr(self.state_dict)


if __name__ == '__main__':
    unittest.main()
