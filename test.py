import unittest

import torch
from torch.utils._benchmark import Timer

from main import get_parser
from models.backbone import build_backbone
from models.criterion import build_criterion
from models.decoder import build_decoder
from models.detr import build_detr
from models.matcher import build_matcher
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
        backbone = build_backbone(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_tensor_list(images)

        self.globals_dict['backbone'] = backbone
        self.globals_dict['images'] = images

        timer = Timer(stmt='backbone(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory backbone (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time backbone (forward): {t: .1f} ms")

    def test_sine_position(self):
        sine_pos_encoder = SinePositionEncoder()

        features = torch.randn(self.args.batch_size, self.args.feat_dim, self.feat_H, self.feat_W, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0

        self.globals_dict['sine_pos_encoder'] = sine_pos_encoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks

        timer = Timer(stmt='sine_pos_encoder(features, feature_masks)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sine pos (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sine pose (forward): {t: .1f} ms")

    def test_encoder(self):
        encoder = build_encoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

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
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

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
        decoder = build_decoder(self.args).to('cuda')

        features = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')
        feature_masks = torch.randn(self.args.batch_size, self.feat_H, self.feat_W, device='cuda') > 0
        pos_encodings = torch.randn(self.feat_H*self.feat_W, self.args.batch_size, self.args.feat_dim, device='cuda')

        self.globals_dict['decoder'] = decoder
        self.globals_dict['features'] = features
        self.globals_dict['feature_masks'] = feature_masks
        self.globals_dict['pos_encodings'] = pos_encodings

        timer = Timer(stmt='decoder(features, feature_masks, pos_encodings)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory sample decoder (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time sample decoder (forward): {t: .1f} ms")

    def test_detr(self):
        detr = build_detr(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_tensor_list(images)

        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images

        timer = Timer(stmt='detr(images)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (forward): {t: .1f} ms")

    def test_matcher(self):
        matcher = build_matcher(self.args).to('cuda')

        num_slots_total = self.args.batch_size * self.args.num_init_slots
        logits = torch.randn(num_slots_total, self.args.num_classes+1, device='cuda')
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda'))
        batch_idx, _ = torch.randint(self.args.batch_size, (num_slots_total,), device='cuda').sort()
        pred_dict = {'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 0}

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['matcher'] = matcher
        self.globals_dict['pred_dict'] = pred_dict
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='matcher(pred_dict, tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory matcher (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time matcher (forward): {t: .1f} ms")

    def test_criterion(self):
        criterion = build_criterion(self.args).to('cuda')

        num_slots_total = self.args.batch_size * self.args.num_init_slots
        logits = torch.randn(num_slots_total, self.args.num_classes+1, device='cuda', requires_grad=True)
        boxes = torch.abs(torch.randn(num_slots_total, 4, device='cuda', requires_grad=True))
        batch_idx, _ = torch.randint(self.args.batch_size, (num_slots_total,), device='cuda').sort()
        pred_list = [{'logits': logits, 'boxes': boxes, 'batch_idx': batch_idx, 'layer_id': 0}]

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['criterion'] = criterion
        self.globals_dict['pred_list'] = pred_list
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='criterion(pred_list, tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory criterion (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time criterion (forward): {t: .1f} ms")

    def test_detr_with_criterion(self):
        detr = build_detr(self.args).to('cuda')
        criterion = build_criterion(self.args).to('cuda')

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W, device='cuda')
        images = nested_tensor_from_tensor_list(images)

        num_target_boxes_total = 20
        labels = torch.randint(self.args.num_classes, (num_target_boxes_total,), device='cuda')
        boxes = torch.abs(torch.randn(num_target_boxes_total, 4, device='cuda'))
        sizes, _ = torch.randint(num_target_boxes_total, (self.args.batch_size+1,), device='cuda').sort()
        tgt_dict = {'labels': labels, 'boxes': boxes, 'sizes': sizes}

        self.globals_dict['detr'] = detr
        self.globals_dict['criterion'] = criterion
        self.globals_dict['images'] = images
        self.globals_dict['tgt_dict'] = tgt_dict

        timer = Timer(stmt='criterion(detr(images), tgt_dict)', globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR with criterion (forward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR with criterion (forward): {t: .1f} ms")


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
        self.args.num_encoder_layers = 1
        self.args.num_decoder_layers = 1

        images = torch.randn(self.args.batch_size, 3, self.pixel_H, self.pixel_W)
        images = nested_tensor_from_tensor_list(images).to('cuda')
        detr = build_detr(self.args).to('cuda')

        stmt = 'torch.cat([v for k,v in detr(images)[0].items() if k in keys], dim=1).sum().backward()'
        self.globals_dict['detr'] = detr
        self.globals_dict['images'] = images
        self.globals_dict['keys'] = ['logits', 'boxes']

        timer = Timer(stmt=stmt, globals=self.globals_dict)
        t = timer.timeit(number=1)._median*1e3

        print(f"Memory DETR model (backward): {torch.cuda.max_memory_allocated()/(1024*1024): .0f} MB")
        print(f"Time DETR model (backward): {t: .1f} ms")

    def test_matcher(self):
        return

    def test_criterion(self):
        return

    def test_detr_with_criterion(self):
        return


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
